Of course. Sharing common data globally is a fundamental requirement for most apps, and Inertia.js provides an elegant, server-driven solution for this.

### **05 - Sharing Global Data with Inertia's Server-Side Sharing**

The key is using the **`HandleInertiaRequests`** middleware that was published during the initial setup. This middleware allows you to share data with *every* Inertia response.

---

#### **Step 1: Locate the Middleware**

The middleware is typically found at `app/Http/Middleware/HandleInertiaRequests.php`. This is where you define the global props.

---

#### **Step 2: Share Data via the `share` Method**

You define a `share` method in this middleware. This method returns an array of props that will be automatically included with every Inertia response.

**Example: Sharing User Data and Flash Messages**

```php
<?php
// app/Http/Middleware/HandleInertiaRequests.php

namespace App\Http\Middleware;

use Illuminate\Http\Request;
use Inertia\Middleware;
use Tightenco\Ziggy\Ziggy; // Optional: for sharing Ziggy routes

class HandleInertiaRequests extends Middleware
{
    // ... other methods (version, etc.) ...

    public function share(Request $request): array
    {
        return array_merge(parent::share($request), [
            // Always share the authenticated user (if available)
            'auth' => [
                'user' => $request->user() ? : null,
            ],
            // Share flash messages from the session
            'flash' => [
                'success' => fn () => $request->session()->get('success'),
                'error' => fn () => $request->session()->get('error'),
                'warning' => fn () => $request->session()->get('warning'),
                'info' => fn () => $request->session()->get('info'),
            ],
            // Optional: Share global app data (e.g., settings from database)
            'app' => [
                'name' => config('app.name'),
                'env' => config('app.env'),
            ],
            // Optional: Share Ziggy routes for frontend use
            'ziggy' => function () use ($request) {
                return array_merge((new Ziggy)->toArray(), [
                    'location' => $request->url(),
                ]);
            },
        ]);
    }
}
```

**Explanation of Shared Props:**

*   **`auth.user`**: The currently authenticated user object. It will be `null` for guests. This is incredibly powerful for conditionally rendering UI.
*   **`flash`**: A collection of flash messages. Using `fn () =>` (a closure) ensures the value is lazily evaluated only when the middleware is called, preventing errors if the session is not set.
*   **`app`**: Any global application data you might need on the frontend.
*   **`ziggy`**: If you have the `ziggy` package installed, this shares your Laravel routes for use in JavaScript.

---

#### **Step 3: Using the Shared Data in Vue Components**

The shared data is available as **props** on the built-in `$page` object, which is accessible in every Vue component.

**1. Accessing the Authenticated User:**
You can use this to show/hide navigation items, display the user's name, or check permissions.

```vue
<script setup>
import { computed } from 'vue';
import { usePage } from '@inertiajs/vue3';

// Access the $page props reactively
const page = usePage();

// The user is available at `page.props.auth.user`
const user = computed(() => page.props.auth.user);

// A computed property to check if a user is logged in
const isAuthenticated = computed(() => !!page.props.auth.user);
</script>

<template>
  <nav>
    <div>
      <Link :href="route('home')">Home</Link>
      <Link :href="route('about')">About</Link>

      <!-- Show these links only to guests -->
      <div v-if="!isAuthenticated">
        <Link :href="route('login')">Login</Link>
        <Link :href="route('register')">Register</Link>
      </div>

      <!-- Show this dropdown only to logged-in users -->
      <div v-else class="flex items-center">
        <span>Hello, {{ user.name }}!</span>
        <Link :href="route('dashboard')">Dashboard</Link>
        <Link :href="route('logout')" method="post">Logout</Link>
      </div>
    </div>
  </nav>
</template>
```

**2. Displaying Flash Messages:**
Create a dedicated component to listen for and display flash messages. This component can be included in your main layout.

**`resources/js/Components/FlashMessages.vue`**
```vue
<script setup>
import { computed } from 'vue';
import { usePage } from '@inertiajs/vue3';

const page = usePage();

// Create a computed property for the flash message
const flash = computed(() => page.props.flash);

// Watch for changes to the flash message and hide it after a delay
import { watch } from 'vue';
import { ref } from 'vue';

const show = ref(false);

watch(flash, (newValue) => {
  // If there's any flash message, show the alert
  if (newValue.success || newValue.error || newValue.warning || newValue.info) {
    show.value = true;
    // Auto-hide after 5 seconds
    setTimeout(() => {
      show.value = false;
    }, 5000);
  }
}, { deep: true }); // 'deep: true' is needed to watch for nested changes
</script>

<template>
  <!-- Success Message -->
  <div v-if="show && flash.success" class="p-4 bg-green-100 text-green-700 rounded mb-4">
    {{ flash.success }}
  </div>

  <!-- Error Message -->
  <div v-if="show && flash.error" class="p-4 bg-red-100 text-red-700 rounded mb-4">
    {{ flash.error }}
  </div>

  <!-- Warning Message -->
  <div v-if="show && flash.warning" class="p-4 bg-yellow-100 text-yellow-700 rounded mb-4">
    {{ flash.warning }}
  </div>

  <!-- Info Message -->
  <div v-if="show && flash.info" class="p-4 bg-blue-100 text-blue-700 rounded mb-4">
    {{ flash.info }}
  </div>
</template>
```

**Include this component in your layout (`AuthenticatedLayout.vue` or `app.blade.php` equivalent):**
```vue
<script setup>
import FlashMessages from '@/Components/FlashMessages.vue';
</script>

<template>
  <div>
    <nav>...</nav>
    <main>
      <!-- Flash messages will appear here -->
      <FlashMessages />
      <slot /> <!-- Your page content gets injected here -->
    </main>
  </div>
</template>
```

---

#### **Step 4: Setting Flash Messages in Laravel Controllers**

You trigger these flash messages from your Laravel controllers using the session.

**Example Controller Methods:**
```php
<?php
// app/Http/Controllers/TaskController.php

namespace App\Http\Controllers;

use App\Models\Task;
use Illuminate\Http\Request;
use Inertia\Inertia;

class TaskController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([...]);

        Task::create($validated);

        // Set a flash message on the session
        return redirect()->route('tasks.index')->with('success', 'Task created successfully!');
    }

    public function update(Request $request, Task $task)
    {
        $validated = $request->validate([...]);

        $task->update($validated);

        return redirect()->back()->with('info', 'Task updated.');
    }

    public function destroy(Task $task)
    {
        $task->delete();

        // You can use different types
        return redirect()->route('tasks.index')->with('error', 'Task deleted permanently.');
    }
}
```

**Key Points:**
*   **`->with('key', 'value')`**: This is the standard Laravel way to flash data to the session.
*   The keys (`'success'`, `'error'`, etc.) must match the keys you defined in the `HandleInertiaRequests` middleware.
*   The shared data is **reactive**. If you navigate to a new page that sets a flash message, the `FlashMessages` component will automatically react and show it.

This pattern keeps your controllers clean and provides a consistent, user-friendly way to give feedback across your entire application.