I'll show you how to set up real-time features with Laravel Echo, broadcasting events from Laravel and listening to them in Vue components.

## 1. Installation and Configuration

### Backend Setup

```bash
# Install Laravel Echo Server (for Pusher alternative)
npm install -g laravel-echo-server

# Or install Pusher PHP SDK
composer require pusher/pusher-php-server

# Install Laravel Echo
npm install laravel-echo pusher-js
```

### Configure Environment (.env)

```env
BROADCAST_DRIVER=pusher
# or for local development with Laravel Echo Server:
# BROADCAST_DRIVER=redis

PUSHER_APP_ID=your-app-id
PUSHER_APP_KEY=your-app-key
PUSHER_APP_SECRET=your-app-secret
PUSHER_APP_CLUSTER=mt1

# For Redis (if using Laravel Echo Server)
REDIS_CLIENT=predis
REDIS_HOST=127.0.0.1
REDIS_PASSWORD=null
REDIS_PORT=6379
```

### Configure Broadcasting (config/broadcasting.php)

```php
'connections' => [
    'pusher' => [
        'driver' => 'pusher',
        'key' => env('PUSHER_APP_KEY'),
        'secret' => env('PUSHER_APP_SECRET'),
        'app_id' => env('PUSHER_APP_ID'),
        'options' => [
            'cluster' => env('PUSHER_APP_CLUSTER'),
            'useTLS' => true,
            'encrypted' => true,
        ],
    ],
],
```

### Bootstrap Laravel Echo (resources/js/echo.js)

```javascript
import Echo from 'laravel-echo';
import Pusher from 'pusher-js';

window.Pusher = Pusher;

window.Echo = new Echo({
    broadcaster: 'pusher',
    key: import.meta.env.VITE_PUSHER_APP_KEY,
    cluster: import.meta.env.VITE_PUSHER_APP_CLUSTER,
    forceTLS: true,
    encrypted: true,
    authorizer: (channel, options) => {
        return {
            authorize: (socketId, callback) => {
                axios.post('/broadcasting/auth', {
                    socket_id: socketId,
                    channel_name: channel.name
                })
                .then(response => {
                    callback(false, response.data);
                })
                .catch(error => {
                    callback(true, error);
                });
            }
        };
    },
});

// For local development with Laravel Echo Server
// window.Echo = new Echo({
//     broadcaster: 'socket.io',
//     host: window.location.hostname + ':6001',
//     auth: {
//         headers: {
//             'Authorization': 'Bearer ' + localStorage.getItem('auth_token')
//         }
//     }
// });

export default window.Echo;
```

### Import Echo in your app (resources/js/app.js)

```javascript
import './echo';
```

## 2. Laravel Event Broadcasting

### Create a Broadcast Event

```bash
php artisan make:event UserStatusUpdated
```

```php
<?php

namespace App\Events;

use App\Models\User;
use Illuminate\Broadcasting\Channel;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Broadcasting\PresenceChannel;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Contracts\Broadcasting\ShouldBroadcast;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class UserStatusUpdated implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $user;
    public $status;
    public $timestamp;

    public function __construct(User $user, string $status)
    {
        $this->user = $user;
        $this->status = $status;
        $this->timestamp = now();
        
        // Don't serialize these properties
        $this->dontBroadcastToCurrentUser();
    }

    public function broadcastOn()
    {
        // Public channel (anyone can listen)
        // return new Channel('user-status');
        
        // Private channel (authenticated users only)
        return new PrivateChannel('user.' . $this->user->id);
        
        // Presence channel (for chat rooms, etc.)
        // return new PresenceChannel('chat-room.1');
    }

    public function broadcastWith()
    {
        return [
            'user' => [
                'id' => $this->user->id,
                'name' => $this->user->name,
                'email' => $this->user->email,
            ],
            'status' => $this->status,
            'updated_at' => $this->timestamp->toISOString(),
        ];
    }

    public function broadcastAs()
    {
        return 'user.status.updated';
    }
}
```

### Channel Authorization (routes/channels.php)

```php
<?php

use App\Models\User;
use Illuminate\Support\Facades\Broadcast;

Broadcast::channel('user.{userId}', function ($user, $userId) {
    return (int) $user->id === (int) $userId;
});

Broadcast::channel('chat-room.{roomId}', function ($user, $roomId) {
    // Check if user has access to this chat room
    return $user->chatRooms()->where('id', $roomId)->exists();
});

Broadcast::channel('notifications.{userId}', function ($user, $userId) {
    return (int) $user->id === (int) $userId;
});
```

### Controller to Trigger Events

```php
<?php

namespace App\Http\Controllers;

use App\Events\UserStatusUpdated;
use App\Models\User;
use Illuminate\Http\Request;

class UserController extends Controller
{
    public function updateStatus(Request $request)
    {
        $request->validate([
            'status' => 'required|in:online,offline,busy,away'
        ]);

        $user = auth()->user();
        $user->update(['status' => $request->status]);

        // Broadcast the event
        broadcast(new UserStatusUpdated($user, $request->status));

        return response()->json([
            'message' => 'Status updated successfully',
            'status' => $request->status
        ]);
    }

    public function sendNotification(Request $request, User $user)
    {
        $request->validate([
            'message' => 'required|string|max:255'
        ]);

        // This would trigger a NotificationSent event
        $user->notify(new CustomNotification($request->message));

        return response()->json(['message' => 'Notification sent']);
    }
}
```

## 3. Vue Components for Real-Time Features

### Real-Time User Status Component

```vue
<template>
  <div class="real-time-status">
    <h3>Real-Time User Status</h3>
    
    <!-- Status Indicator -->
    <div class="status-indicator" :class="status">
      <span class="status-dot"></span>
      {{ statusText }}
    </div>

    <!-- Online Users List -->
    <div v-if="onlineUsers.length > 0" class="online-users">
      <h4>Online Users ({{ onlineUsers.length }})</h4>
      <div v-for="user in onlineUsers" :key="user.id" class="user-item">
        <span class="user-name">{{ user.name }}</span>
        <span class="user-status online">●</span>
      </div>
    </div>

    <!-- Status Update Form -->
    <div class="status-controls">
      <select v-model="selectedStatus" @change="updateStatus">
        <option value="online">Online</option>
        <option value="away">Away</option>
        <option value="busy">Busy</option>
        <option value="offline">Offline</option>
      </select>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { usePage } from '@inertiajs/inertia-vue3'
import axios from 'axios'

const currentUser = usePage().props.value.auth.user
const status = ref(currentUser.status || 'offline')
const selectedStatus = ref(status.value)
const onlineUsers = ref([])

const statusText = computed(() => {
  const texts = {
    online: 'Online',
    away: 'Away',
    busy: 'Busy',
    offline: 'Offline'
  }
  return texts[status.value]
})

// Listen for status updates
let echoListener = null

onMounted(() => {
  initializeEchoListeners()
  fetchOnlineUsers()
})

onUnmounted(() => {
  if (echoListener) {
    echoListener.stopListening()
  }
})

const initializeEchoListeners = () => {
  // Listen for private user status updates
  echoListener = window.Echo.private(`user.${currentUser.id}`)
    .listen('UserStatusUpdated', (event) => {
      status.value = event.status
      selectedStatus.value = event.status
    })

  // Listen for presence channel events (online users)
  window.Echo.join('online-users')
    .here((users) => {
      onlineUsers.value = users
    })
    .joining((user) => {
      onlineUsers.value = [...onlineUsers.value, user]
    })
    .leaving((user) => {
      onlineUsers.value = onlineUsers.value.filter(u => u.id !== user.id)
    })

  // Listen for global notifications
  window.Echo.channel('global-notifications')
    .listen('NotificationSent', (event) => {
      showNotification(event.message)
    })
}

const fetchOnlineUsers = async () => {
  try {
    const response = await axios.get('/api/online-users')
    onlineUsers.value = response.data
  } catch (error) {
    console.error('Failed to fetch online users:', error)
  }
}

const updateStatus = async () => {
  try {
    await axios.post('/api/user/status', {
      status: selectedStatus.value
    })
  } catch (error) {
    console.error('Failed to update status:', error)
  }
}

const showNotification = (message) => {
  // Implement your notification system
  console.log('New notification:', message)
  // You could use a toast library here
}
</script>

<style scoped>
.real-time-status {
  padding: 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.status-indicator.online .status-dot { background-color: #48bb78; }
.status-indicator.away .status-dot { background-color: #ed8936; }
.status-indicator.busy .status-dot { background-color: #f56565; }
.status-indicator.offline .status-dot { background-color: #a0aec0; }

.online-users {
  margin-top: 1rem;
}

.user-item {
  display: flex;
  justify-content: between;
  align-items: center;
  padding: 0.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.user-status.online {
  color: #48bb78;
}

.status-controls {
  margin-top: 1rem;
}

select {
  padding: 0.5rem;
  border: 1px solid #cbd5e0;
  border-radius: 0.25rem;
}
</style>
```

### Real-Time Notification System

```vue
<template>
  <div class="notifications-container">
    <div class="notifications-header">
      <h3>Notifications</h3>
      <span class="badge" :class="{ 'has-unread': hasUnread }">
        {{ notifications.length }}
      </span>
    </div>

    <div class="notifications-list">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        :class="['notification-item', { unread: !notification.read_at }]"
        @click="markAsRead(notification)"
      >
        <div class="notification-content">
          <p class="notification-message">{{ notification.data.message }}</p>
          <span class="notification-time">
            {{ formatTime(notification.created_at) }}
          </span>
        </div>
        <button
          v-if="!notification.read_at"
          @click.stop="markAsRead(notification)"
          class="mark-read-btn"
        >
          Mark read
        </button>
      </div>
    </div>

    <div v-if="notifications.length === 0" class="empty-state">
      No notifications yet
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { usePage } from '@inertiajs/inertia-vue3'
import axios from 'axios'

const currentUser = usePage().props.value.auth.user
const notifications = ref([])

const hasUnread = computed(() => {
  return notifications.value.some(n => !n.read_at)
})

onMounted(() => {
  fetchNotifications()
  listenForNotifications()
})

onUnmounted(() => {
  window.Echo.leave(`notifications.${currentUser.id}`)
})

const fetchNotifications = async () => {
  try {
    const response = await axios.get('/api/notifications')
    notifications.value = response.data
  } catch (error) {
    console.error('Failed to fetch notifications:', error)
  }
}

const listenForNotifications = () => {
  window.Echo.private(`notifications.${currentUser.id}`)
    .listen('NotificationSent', (event) => {
      // Add new notification to the list
      notifications.value.unshift({
        id: Date.now(), // temporary ID
        data: { message: event.message },
        read_at: null,
        created_at: new Date().toISOString()
      })
    })
}

const markAsRead = async (notification) => {
  try {
    await axios.post(`/api/notifications/${notification.id}/read`)
    notification.read_at = new Date().toISOString()
  } catch (error) {
    console.error('Failed to mark notification as read:', error)
  }
}

const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString()
}
</script>

<style scoped>
.notifications-container {
  max-width: 400px;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
  padding: 1rem;
}

.notifications-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.badge {
  background-color: #e2e8f0;
  color: #4a5568;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.75rem;
}

.badge.has-unread {
  background-color: #f56565;
  color: white;
}

.notification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  border-bottom: 1px solid #e2e8f0;
  cursor: pointer;
}

.notification-item.unread {
  background-color: #ebf8ff;
}

.notification-item:last-child {
  border-bottom: none;
}

.notification-content {
  flex: 1;
}

.notification-message {
  margin: 0;
  font-size: 0.875rem;
}

.notification-time {
  font-size: 0.75rem;
  color: #718096;
}

.mark-read-btn {
  background: none;
  border: 1px solid #cbd5e0;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  cursor: pointer;
}

.mark-read-btn:hover {
  background-color: #edf2f7;
}

.empty-state {
  text-align: center;
  color: #a0aec0;
  padding: 2rem;
}
</style>
```

### Real-Time Chat Component

```vue
<template>
  <div class="chat-container">
    <div class="chat-header">
      <h3>Chat Room</h3>
      <span class="online-count">{{ usersOnline }} online</span>
    </div>

    <div class="messages-container" ref="messagesContainer">
      <div
        v-for="message in messages"
        :key="message.id"
        :class="['message', { own: message.user_id === currentUser.id }]"
      >
        <div class="message-header">
          <span class="user-name">{{ message.user.name }}</span>
          <span class="message-time">{{ formatTime(message.created_at) }}</span>
        </div>
        <div class="message-content">{{ message.content }}</div>
      </div>
    </div>

    <div class="message-input">
      <input
        v-model="newMessage"
        @keypress.enter="sendMessage"
        placeholder="Type a message..."
        :disabled="!isConnected"
      />
      <button @click="sendMessage" :disabled="!newMessage.trim() || !isConnected">
        Send
      </button>
    </div>

    <div v-if="!isConnected" class="connection-status">
      Connecting to chat...
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { usePage } from '@inertiajs/inertia-vue3'
import axios from 'axios'

const currentUser = usePage().props.value.auth.user
const newMessage = ref('')
const messages = ref([])
const usersOnline = ref(0)
const isConnected = ref(false)
const messagesContainer = ref(null)

onMounted(() => {
  fetchMessages()
  joinChatRoom()
})

onUnmounted(() => {
  leaveChatRoom()
})

const fetchMessages = async () => {
  try {
    const response = await axios.get('/api/chat/messages')
    messages.value = response.data
    scrollToBottom()
  } catch (error) {
    console.error('Failed to fetch messages:', error)
  }
}

const joinChatRoom = () => {
  window.Echo.join('chat-room.1')
    .here((users) => {
      usersOnline.value = users.length
      isConnected.value = true
    })
    .joining((user) => {
      usersOnline.value += 1
      showSystemMessage(`${user.name} joined the chat`)
    })
    .leaving((user) => {
      usersOnline.value -= 1
      showSystemMessage(`${user.name} left the chat`)
    })
    .listen('MessageSent', (event) => {
      messages.value.push(event.message)
      scrollToBottom()
    })
    .error((error) => {
      console.error('Chat connection error:', error)
      isConnected.value = false
    })
}

const leaveChatRoom = () => {
  window.Echo.leave('chat-room.1')
}

const sendMessage = async () => {
  if (!newMessage.value.trim()) return

  try {
    await axios.post('/api/chat/messages', {
      content: newMessage.value.trim()
    })
    newMessage.value = ''
  } catch (error) {
    console.error('Failed to send message:', error)
  }
}

const showSystemMessage = (text) => {
  messages.value.push({
    id: Date.now(),
    content: text,
    user: { name: 'System' },
    created_at: new Date().toISOString(),
    is_system: true
  })
  scrollToBottom()
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 500px;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
}

.chat-header {
  padding: 1rem;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #f7fafc;
}

.online-count {
  color: #48bb78;
  font-size: 0.875rem;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.message {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border-radius: 0.5rem;
  background-color: #f7fafc;
}

.message.own {
  background-color: #ebf8ff;
  margin-left: 2rem;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
}

.user-name {
  font-weight: 600;
  font-size: 0.875rem;
}

.message-time {
  font-size: 0.75rem;
  color: #718096;
}

.message-content {
  font-size: 0.875rem;
}

.message-input {
  padding: 1rem;
  border-top: 1px solid #e2e8f0;
  display: flex;
  gap: 0.5rem;
}

.message-input input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #cbd5e0;
  border-radius: 0.25rem;
}

.message-input button {
  padding: 0.5rem 1rem;
  background-color: #4299e1;
  color: white;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
}

.message-input button:disabled {
  background-color: #cbd5e0;
  cursor: not-allowed;
}

.connection-status {
  padding: 0.5rem;
  text-align: center;
  background-color: #fed7d7;
  color: #c53030;
}
</style>
```

## 4. Additional Event Examples

### Notification Event

```php
<?php

namespace App\Events;

use App\Models\User;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Contracts\Broadcasting\ShouldBroadcast;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class NotificationSent implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $userId;
    public $message;

    public function __construct($userId, $message)
    {
        $this->userId = $userId;
        $this->message = $message;
    }

    public function broadcastOn()
    {
        return new PrivateChannel('notifications.' . $this->userId);
    }

    public function broadcastWith()
    {
        return [
            'message' => $this->message,
            'timestamp' => now()->toISOString(),
        ];
    }
}
```

### Message Event

```php
<?php

namespace App\Events;

use App\Models\Message;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Broadcasting\PresenceChannel;
use Illuminate\Contracts\Broadcasting\ShouldBroadcast;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class MessageSent implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $message;

    public function __construct(Message $message)
    {
        $this->message = $message;
    }

    public function broadcastOn()
    {
        return new PresenceChannel('chat-room.' . $this->message->room_id);
    }

    public function broadcastWith()
    {
        return [
            'message' => [
                'id' => $this->message->id,
                'content' => $this->message->content,
                'user_id' => $this->message->user_id,
                'user' => [
                    'id' => $this->message->user->id,
                    'name' => $this->message->user->name,
                ],
                'created_at' => $this->message->created_at->toISOString(),
            ]
        ];
    }
}
```

This setup provides a complete real-time system with:

1. **User status updates** across devices
2. **Real-time notifications** 
3. **Live chat functionality**
4. **Presence channels** for tracking online users
5. **Private channels** for user-specific events
6. **Proper error handling** and connection states

The system scales from simple status updates to complex real-time applications with proper authentication and authorization.