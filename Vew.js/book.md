### **Installing Vue.js 3 in a Laravel Project Using Vite**  

Since Laravel 9+, Vite has replaced Laravel Mix as the default frontend build tool. Here’s how to set up **Vue 3** in a Laravel project with Vite:

---

### **Step 1: Install Laravel (if starting fresh)**
```bash
composer create-project laravel/laravel vue-laravel-app
cd vue-laravel-app
```

---

### **Step 2: Install Vue 3 and Required Dependencies**
```bash
npm install vue@next @vitejs/plugin-vue
```
- `vue@next` → Vue 3  
- `@vitejs/plugin-vue` → Official Vite plugin for Vue  

---

### **Step 3: Configure Vite (`vite.config.js`)**
Update the file in your project root:
```js
import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
    plugins: [
        laravel({
            input: ['resources/css/app.css', 'resources/js/app.js'],
            refresh: true,
        }),
        vue(), // Enable Vue 3 support
    ],
});
```

---

### **Step 4: Update `resources/js/app.js` for Vue**
Replace the default file with:
```js
import { createApp } from 'vue';
import App from './App.vue';

createApp(App).mount('#app');
```

---

### **Step 5: Create a Vue Component (`resources/js/App.vue`)**
```vue
<template>
    <h1>Hello Vue 3 + Laravel!</h1>
    <p>This is a Vue component.</p>
</template>

<script>
export default {
    name: 'App',
};
</script>
```

---

### **Step 6: Update Blade File to Load Vue (`resources/views/welcome.blade.php`)**
Replace the default view with:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Laravel + Vue 3</title>
    @vite(['resources/js/app.js'])
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

---

### **Step 7: Run the Development Server**
```bash
npm run dev
```
Visit `http://localhost:8000` to see Vue 3 working with Laravel!  

---

### **Bonus: Auto-Refresh & Production Build**
- **Hot reloading** works automatically with `npm run dev`.  
- For production:  
  ```bash
  npm run build
  ```

---

### **Troubleshooting**
- **"Vue is not defined"** → Ensure `@vitejs/plugin-vue` is installed and configured.  
- **"Failed to resolve Vue"** → Check `app.js` imports and `vite.config.js`.  

---
### **Vue's Options API vs. Composition API: Key Differences**  

Vue 3 supports two ways of writing components:  

1. **Options API** (Vue 2 style)  
2. **Composition API** (Vue 3’s modern approach)  

Here’s a breakdown of their differences and which one to use with Laravel.  

---

## **1. Options API (Classic Vue Style)**  
**How it works:**  
- Organizes code by **options** (`data`, `methods`, `computed`, `lifecycle hooks`, etc.).  
- Best for **smaller components** or developers coming from Vue 2.  

**Example:**  
```vue
<script>
export default {
    data() {
        return {
            count: 0,
        };
    },
    methods: {
        increment() {
            this.count++;
        },
    },
    mounted() {
        console.log('Component mounted!');
    },
};
</script>
```

✅ **Pros:**  
- Easier for beginners (structured, self-contained).  
- Familiar for Vue 2 developers.  

❌ **Cons:**  
- Logic is **scattered** (e.g., a feature’s `data`, `methods`, and `computed` are separated).  
- Harder to reuse logic across components.  

---

## **2. Composition API (Modern Vue 3 Style)**  
**How it works:**  
- Uses `setup()` function to group related logic.  
- Encourages **reusability** (via composables).  
- Better for **larger apps** or complex logic.  

**Example:**  
```vue
<script>
import { ref, onMounted } from 'vue';

export default {
    setup() {
        const count = ref(0); // Reactive data

        const increment = () => {
            count.value++;
        };

        onMounted(() => {
            console.log('Component mounted!');
        });

        return { count, increment }; // Expose to template
    },
};
</script>
```

✅ **Pros:**  
- **Better code organization** (logic grouped by feature, not by type).  
- **Easier code reuse** (via composables, like `useUserAuth()`).  
- **Better TypeScript support**.  

❌ **Cons:**  
- Steeper learning curve (requires understanding `ref`, `reactive`, etc.).  
- More verbose for simple components.  

---

## **Which One Should You Use with Laravel?**  
| Scenario | Recommended API |
|-----------|----------------|
| **✅ You're new to Vue** | **Options API** (easier to learn) |  
| **✅ Small Laravel apps (simple CRUDs)** | **Options API** (less boilerplate) |  
| **✅ Large Laravel apps (SPAs, complex logic)** | **Composition API** (better scalability) |  
| **✅ Using TypeScript** | **Composition API** (better type inference) |  
| **✅ Reusing logic (e.g., auth, API calls)** | **Composition API** (via composables) |  

### **Recommendation for Laravel Developers**  
- **Start with Options API** if you’re learning Vue.  
- **Switch to Composition API** when:  
  - Your app grows in complexity.  
  - You need reusable logic (e.g., API fetching, form handling).  
  - You want better TypeScript support.  

---

## **Bonus: Using Composition API with `<script setup>` (Even Simpler!)**  
Vue 3.2+ introduced **`<script setup>`**, a cleaner way to use the Composition API:  

```vue
<script setup>
import { ref, onMounted } from 'vue';

const count = ref(0);

const increment = () => {
    count.value++;
};

onMounted(() => {
    console.log('Component mounted!');
});
</script>
```
- **No need for `setup()` function** – variables auto-expose to template.  
- **Best for modern Laravel + Vue apps.**  

---
### **Creating & Rendering a Vue Component in Laravel (Step-by-Step)**  

Since you're using **Vue 3 with Laravel (Vite)**, here’s how to create a simple Vue component and embed it in a Blade view.

---

## **Step 1: Create a Vue Component**
### **1.1. Create a new Vue component file**  
Location: `resources/js/components/ExampleComponent.vue`  
```vue
<template>
    <div class="bg-blue-100 p-4 rounded-lg">
        <h2 class="text-xl font-bold">{{ title }}</h2>
        <p>{{ message }}</p>
        <button 
            @click="count++" 
            class="mt-2 px-4 py-2 bg-blue-500 text-white rounded"
        >
            Clicked {{ count }} times
        </button>
    </div>
</template>

<script setup>
import { ref } from 'vue';

// Reactive data
const count = ref(0);
const title = "Hello from Vue!";
const message = "This component is rendered inside Laravel Blade.";
</script>
```
- Uses **`<script setup>`** (modern Composition API syntax).  
- Has reactive data (`count`), a title, and a button.  

---

### **1.2. Register the Component (Optional)**
If you want to **globally register** it (so it’s available everywhere), modify `resources/js/app.js`:
```js
import { createApp } from 'vue';
import ExampleComponent from './components/ExampleComponent.vue';

const app = createApp({});
app.component('example-component', ExampleComponent); // Global registration
app.mount('#app');
```
*(Skip this if you prefer **local component imports**.)*

---

## **Step 2: Render the Vue Component in Blade**
### **2.1. Update `resources/views/welcome.blade.php`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Laravel + Vue</title>
    @vite(['resources/js/app.js']) <!-- Loads Vue & your components -->
</head>
<body>
    <div class="p-10">
        <!-- Laravel Blade content -->
        <h1 class="text-3xl mb-6">Laravel Blade View</h1>

        <!-- Vue Component -->
        <div id="app">
            <example-component></example-component>
        </div>
    </div>
</body>
</html>
```
- `@vite` loads the compiled JavaScript.  
- `<example-component>` is rendered where you place it.  

---

### **2.2. Alternative: Dynamic Props from Laravel**
If you want to pass **Laravel data** to Vue, use `props`:
```blade
<example-component :initial-count="{{ json_encode(5) }}"></example-component>
```
Then modify `ExampleComponent.vue`:
```vue
<script setup>
const props = defineProps({
    initialCount: { type: Number, default: 0 }
});

const count = ref(props.initialCount); // Now uses Laravel-passed data
</script>
```

---

## **Step 3: Run the App**
```bash
npm run dev
```
Visit `http://localhost:8000` to see:  
- Laravel Blade renders the Vue component.  
- The Vue component is **reactive** (button clicks update `count`).  

---

## **Key Takeaways**
1. **Vue components** live in `/resources/js/components/`.  
2. **Global registration** (optional) in `app.js`.  
3. **Blade renders Vue** via `@vite` and `<example-component>`.  
4. **Pass Laravel data** using `props`.  

---
# Passing Data from Laravel (Blade) to Vue Components Using Props

The cleanest way to pass data from Laravel to Vue components is by using props. Here's a complete guide with best practices:

## Method 1: Direct Prop Binding (Simple Values)

### 1. In your Blade template:
```php
<example-component 
    :user-id="{{ auth()->id() }}"
    :user-name="{{ json_encode(auth()->user()->name) }}"
    :is-admin="{{ auth()->user()->isAdmin ? 'true' : 'false' }}"
></example-component>
```

### 2. In your Vue component:
```vue
<script setup>
const props = defineProps({
    userId: {
        type: Number,
        required: true
    },
    userName: {
        type: String,
        default: 'Guest'
    },
    isAdmin: {
        type: Boolean,
        default: false
    }
});
</script>
```

## Method 2: Passing Complex Data (Arrays/Objects)

### Blade template:
```php
<example-component 
    :initial-data="{{ json_encode([
        'user' => auth()->user(),
        'settings' => $settings,
        'csrf_token' => csrf_token()
    ]) }}"
></example-component>
```

### Vue component:
```vue
<script setup>
const props = defineProps({
    initialData: {
        type: Object,
        required: true
    }
});

// Access data
const { user, settings, csrf_token } = props.initialData;
</script>
```

## Method 3: Using Inertia.js (For SPAs)

If you're building an SPA with Inertia.js:

```php
return Inertia::render('Dashboard', [
    'users' => User::all(),
    'stats' => $stats,
]);
```

Then access directly in Vue:
```javascript
const { users, stats } = usePage().props;
```

## Best Practices & Gotchas

1. **Always escape JSON properly**:
   ```php
   :items="{{ json_encode($data, JSON_HEX_TAG) }}"
   ```

2. **For large datasets**, consider:
   - API endpoints + Axios fetching
   - Laravel pagination with Vue rendering

3. **Security considerations**:
   - Don't pass sensitive data directly
   - Use hidden fields for CSRF tokens

4. **Performance tip**:
   ```php
   :compact-data="{{ Js::from($minifiedData) }}"
   ```

## Complete Example

**Blade:**
```php
<user-profile 
    :user="{{ Js::from([
        'id' => $user->id,
        'name' => $user->name,
        'avatar' => $user->avatar_url,
        'joined_at' => $user->created_at->diffForHumans()
    ]) }}"
    :stats="{{ Js::from($user->stats) }}"
></user-profile>
```

**Vue Component:**
```vue
<template>
  <div>
    <img :src="user.avatar" :alt="user.name">
    <h2>{{ user.name }}</h2>
    <p>Member since {{ user.joined_at }}</p>
    <div v-for="stat in stats" :key="stat.id">
      {{ stat.label }}: {{ stat.value }}
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  user: {
    type: Object,
    required: true
  },
  stats: {
    type: Array,
    default: () => []
  }
});
</script>
```
### **Handling Form Submissions in Vue.js with Laravel CSRF Protection**

When submitting forms from Vue.js to Laravel, you need to:
1. Include Laravel's CSRF token
2. Handle form data properly
3. Manage validation errors

Here's the **best way** to do it:

---

## **Method 1: Using Axios (Recommended)**
### **1. Set Up Axios Globally (resources/js/app.js)**
```js
import axios from 'axios';
window.axios = axios;

// Set CSRF token for all requests
window.axios.defaults.headers.common['X-Requested-With'] = 'XMLHttpRequest';
window.axios.defaults.withCredentials = true;

// Get CSRF token from meta tag (included by default in Laravel)
const token = document.head.querySelector('meta[name="csrf-token"]');
if (token) {
    window.axios.defaults.headers.common['X-CSRF-TOKEN'] = token.content;
}
```

### **2. Create a Vue Form Component**
```vue
<template>
  <form @submit.prevent="submitForm">
    <input v-model="form.name" type="text" placeholder="Name">
    <input v-model="form.email" type="email" placeholder="Email">
    <button type="submit">Submit</button>
  </form>
  <div v-if="errors">
    <p v-for="error in errors" :key="error" class="text-red-500">{{ error }}</p>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue';

const form = reactive({
  name: '',
  email: '',
});

const errors = ref({});

const submitForm = async () => {
  try {
    const response = await axios.post('/api/submit-form', form);
    console.log(response.data);
    // Reset form on success
    form.name = '';
    form.email = '';
    errors.value = {};
  } catch (error) {
    if (error.response.status === 422) {
      errors.value = error.response.data.errors;
    }
  }
};
</script>
```

---

## **Method 2: Using Laravel's Built-in CSRF (For Blade + Vue Hybrid)**
### **1. Include CSRF Token in Blade**
```html
<form @submit.prevent="submitForm">
  @csrf <!-- Laravel adds hidden CSRF field -->
  <input v-model="form.name" type="text">
  <button type="submit">Submit</button>
</form>
```

### **2. Submit Using `FormData` (for file uploads)**
```js
const submitForm = async () => {
  const formData = new FormData();
  formData.append('name', form.name);
  formData.append('avatar', fileInput.files[0]);

  const response = await axios.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};
```

---

## **Method 3: Using Laravel Sanctum (For SPAs)**
If your Vue app is on a different domain:

### **1. Install Sanctum**
```bash
composer require laravel/sanctum
php artisan vendor:publish --provider="Laravel\Sanctum\SanctumServiceProvider"
```

### **2. Configure CORS (config/cors.php)**
```php
'paths' => ['api/*', 'sanctum/csrf-cookie'],
'allowed_methods' => ['*'],
'allowed_origins' => ['http://your-vue-app.com'],
```

### **3. Fetch CSRF Cookie First**
```js
await axios.get('/sanctum/csrf-cookie');
await axios.post('/api/submit', form);
```

---

## **Best Practices**
1. **Always use `@csrf` in Blade forms** (if mixing Blade + Vue)
2. **For pure Vue apps**, attach CSRF token via Axios defaults
3. **Handle 422 validation errors** gracefully
4. **For file uploads**, use `FormData`
5. **For SPAs**, use Sanctum for CSRF + API auth

---

## **Complete Laravel Backend Example**
```php
// routes/api.php
Route::post('/submit-form', function (Request $request) {
  $validated = $request->validate([
    'name' => 'required|string|max:255',
    'email' => 'required|email',
  ]);

  // Process data...
  return response()->json(['success' => true]);
});
```
# Implementing Vue Router in Laravel for an SPA Experience

Here's a comprehensive guide to setting up Vue Router in a Laravel project to create a single-page application (SPA) while using Laravel as the backend API:

## 1. Install Vue Router

```bash
npm install vue-router@4
```

## 2. Configure Vue Router

Create a router file at `resources/js/router/index.js`:

```javascript
import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import About from '../views/About.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/about',
    name: 'About',
    component: About
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
```

## 3. Modify Your Main App File

Update `resources/js/app.js`:

```javascript
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

createApp(App)
  .use(router)
  .mount('#app')
```

## 4. Create the App.vue Component

```vue
<template>
  <div id="app">
    <nav>
      <router-link to="/">Home</router-link> |
      <router-link to="/about">About</router-link>
    </nav>
    <router-view/>
  </div>
</template>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  text-align: center;
  color: #2c3e50;
}
nav {
  padding: 30px;
}
nav a {
  font-weight: bold;
  color: #2c3e50;
}
nav a.router-link-exact-active {
  color: #42b983;
}
</style>
```

## 5. Set Up Laravel to Handle Routing

Create a catch-all route in `routes/web.php`:

```php
Route::get('/{any}', function () {
    return view('app');
})->where('any', '.*');
```

## 6. Create the Blade Template

Create `resources/views/app.blade.php`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Laravel + Vue SPA</title>
    @vite(['resources/js/app.js'])
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

## 7. Create Sample Views

Example view at `resources/js/views/Home.vue`:

```vue
<template>
  <div class="home">
    <h1>Welcome to the Home Page</h1>
  </div>
</template>
```

## 8. Configure Vite

Update `vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import laravel from 'laravel-vite-plugin'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [
    laravel({
      input: ['resources/js/app.js'],
      refresh: true,
    }),
    vue({
      template: {
        transformAssetUrls: {
          base: null,
          includeAbsolute: false,
        },
      },
    }),
  ],
})
```

## 9. API Integration

For API calls, create an `axios` instance in `resources/js/api.js`:

```javascript
import axios from 'axios'

export default axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest'
  }
})
```

## 10. Navigation Guards (Optional)

Add authentication check in router:

```javascript
router.beforeEach((to, from, next) => {
  if (to.matched.some(record => record.meta.requiresAuth)) {
    if (!localStorage.getItem('authToken')) {
      next({ name: 'Login' })
    } else {
      next()
    }
  } else {
    next()
  }
})
```

## Key Considerations:

1. **Server Configuration**: Ensure your server routes all requests to your `app.blade.php` view
2. **SEO**: Consider SSR with Inertia.js or Nuxt.js if SEO is critical
3. **Authentication**: Use Laravel Sanctum for API authentication
4. **Code Splitting**: Implement lazy loading for better performance:

```javascript
const routes = [
  {
    path: '/about',
    name: 'About',
    component: () => import('../views/About.vue')
  }
]
```
### **Pinia: Modern State Management for Laravel + Vue Apps**

Pinia is Vue's **official state management library** (replacing Vuex in Vue 3). It offers:
- Simpler syntax than Vuex  
- Full TypeScript support  
- DevTools integration  
- Modular by design  

Perfect for managing **global state** (user auth, API data, UI state) in Laravel+Vue apps.

---

## **1. Install Pinia**
```bash
npm install pinia
```

---

## **2. Set Up Pinia in Laravel**
### **2.1. Initialize Pinia (`resources/js/app.js`)**
```js
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';

const app = createApp(App);
app.use(createPinia());
app.mount('#app');
```

---

## **3. Create Your First Store**
### **Example: Auth Store (`resources/js/stores/auth.js`)**
```js
import { defineStore } from 'pinia';
import axios from 'axios';

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    token: localStorage.getItem('token') || null,
  }),

  actions: {
    async login(credentials) {
      const response = await axios.post('/api/login', credentials);
      this.user = response.data.user;
      this.token = response.data.token;
      localStorage.setItem('token', response.data.token);
    },

    logout() {
      this.user = null;
      this.token = null;
      localStorage.removeItem('token');
    }
  },

  getters: {
    isAuthenticated: (state) => !!state.token,
  },
});
```

---

## **4. Use the Store in Components**
### **Login Component Example (`resources/js/components/Login.vue`)**
```vue
<script setup>
import { ref } from 'vue';
import { useAuthStore } from '@/stores/auth';

const auth = useAuthStore();
const form = ref({ email: '', password: '' });

const handleLogin = async () => {
  await auth.login(form.value);
  // Redirect after login
  router.push('/dashboard');
};
</script>

<template>
  <form @submit.prevent="handleLogin">
    <input v-model="form.email" type="email">
    <input v-model="form.password" type="password">
    <button type="submit">Login</button>
  </form>
</template>
```

---

## **5. Access State in Other Components**
### **Navbar Component (`resources/js/components/Navbar.vue`)**
```vue
<script setup>
import { useAuthStore } from '@/stores/auth';
const auth = useAuthStore();
</script>

<template>
  <div>
    <span v-if="auth.isAuthenticated">
      Welcome, {{ auth.user.name }}!
    </span>
    <button v-if="auth.isAuthenticated" @click="auth.logout">
      Logout
    </button>
  </div>
</template>
```

---

## **6. Hydrate Store with Laravel Data**
### **Pass Initial State from Blade**
```php
<main-app :initial-auth="{{ json_encode([
    'user' => auth()->user(),
    'token' => session('token'),
]) }}"></main-app>
```

### **Hydrate Pinia Store on Load**
```js
// In your main App.vue
import { useAuthStore } from '@/stores/auth';

const props = defineProps(['initialAuth']);
const auth = useAuthStore();

if (props.initialAuth) {
  auth.$patch({
    user: props.initialAuth.user,
    token: props.initialAuth.token,
  });
}
```

---

## **Key Benefits for Laravel Apps**
1. **Shared State**  
   - Access user/auth state across all components  
2. **API Caching**  
   - Store API responses to avoid duplicate requests  
3. **Persistent State**  
   - Sync with localStorage for page refreshes  
4. **Type Safety**  
   - Works perfectly with TypeScript  

---

## **Pinia vs. Vuex**
| Feature         | Pinia | Vuex |
|----------------|-------|------|
| Syntax         | Simpler | More boilerplate |
| TypeScript     | Excellent | Good |
| Modules        | Automatic | Manual |
| DevTools       | Built-in | Plugin |
| Size           | 1KB | 4KB |

---

## **Advanced Pattern: API Service Integration**
```js
// resources/js/stores/posts.js
export const usePostStore = defineStore('posts', {
  state: () => ({
    posts: [],
  }),

  actions: {
    async fetchPosts() {
      this.posts = await axios.get('/api/posts');
    },
  },
});
```

---

### **When to Use Pinia in Laravel**
1. **User authentication state**  
2. **Shared UI state (modals, themes)**  
3. **Caching API responses**  
4. **Complex multi-component state**  

For simple props, just use component state. For everything else: **Pinia**.

---

### Fetching Data from Laravel API in Vue.js with Axios

Here's a complete, production-ready approach to consuming Laravel API endpoints in Vue:

#### 1. First, set up Axios globally (recommended):

```javascript
// resources/js/axios.js
import axios from 'axios';

const api = axios.create({
  baseURL: '/api', // Your Laravel API prefix
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth tokens
api.interceptors.request.use(config => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response.status === 401) {
      // Handle unauthorized
      router.push('/login');
    }
    return Promise.reject(error);
  }
);

export default api;
```

#### 2. Create a composable for API calls (modern approach):

```javascript
// resources/js/composables/useApi.js
import { ref } from 'vue';
import api from '@/axios';

export function useApi() {
  const data = ref(null);
  const loading = ref(false);
  const error = ref(null);

  const fetchData = async (url, config = {}) => {
    loading.value = true;
    error.value = null;
    try {
      const response = await api.get(url, config);
      data.value = response.data;
    } catch (err) {
      error.value = err.response?.data?.message || err.message;
    } finally {
      loading.value = false;
    }
  };

  return { data, loading, error, fetchData };
}
```

#### 3. Use in components with reactive state:

```vue
<script setup>
import { useApi } from '@/composables/useApi';

const { data: posts, loading, error, fetchData } = useApi();

// Fetch on component mount
fetchData('/posts');

// Optional: Refetch with filters
const applyFilters = (filters) => {
  fetchData('/posts', { params: filters });
};
</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else-if="error" class="text-red-500">{{ error }}</div>
  <ul v-else>
    <li v-for="post in posts" :key="post.id">
      {{ post.title }}
    </li>
  </ul>
</template>
```

#### 4. For POST/PUT/DELETE operations:

```javascript
// In your useApi composable
const postData = async (url, payload) => {
  loading.value = true;
  try {
    const response = await api.post(url, payload);
    data.value = response.data;
    return response;
  } catch (err) {
    error.value = err.response?.data?.errors || err.message;
    throw err; // Re-throw for form handling
  } finally {
    loading.value = false;
  }
};

// In component:
const handleSubmit = async () => {
  try {
    await postData('/posts', formData);
    // Success handling
  } catch {
    // Error displayed via the composable
  }
};
```

#### 5. Laravel API best practices:

1. **API Resource Classes** (for consistent responses):
```php
// app/Http/Resources/PostResource.php
public function toArray($request)
{
  return [
    'id' => $this->id,
    'title' => $this->title,
    'created_at' => $this->created_at->diffForHumans()
  ];
}
```

2. **Pagination**:
```javascript
fetchData('/posts?page=2');
```
```php
// Controller
return PostResource::collection(Post::paginate(15));
```

3. **Error Handling**:
```php
// Handler.php
public function render($request, Throwable $e)
{
  if ($request->expectsJson()) {
    return response()->json([
      'message' => $e->getMessage(),
      'errors' => $e->errors() // For validation
    ], $this->getStatusCode($e));
  }
  // ...
}
```

#### 6. Advanced: TypeScript Support

```typescript
// types/api.d.ts
interface Post {
  id: number;
  title: string;
  body: string;
  created_at: string;
}

// In composable
const fetchData = async <T>(url: string): Promise<T> => {
  const response = await api.get<T>(url);
  return response.data;
};

// Usage
const posts = await fetchData<Post[]>('/posts');
```

#### Key Takeaways:

1. **Centralize API logic** in axios instance + composables
2. **Handle loading/error states** reactively
3. **Use Laravel Resources** for consistent API responses
4. **Implement proper error handling** for production
5. **TypeScript** adds safety for larger apps

This approach gives you:
- Reusable API logic
- Great TypeScript support
- Production-ready error handling
- Seamless Laravel integration

---
### **Real-Time Updates in Vue.js with Laravel Echo & WebSockets**  
**(Using Pusher, Laravel Broadcasting, and Vue 3)**

Here's a **step-by-step guide** to implement real-time features (e.g., live notifications, chat, dashboard updates) in a Laravel + Vue app.

---

## **1. Install Required Packages**
```bash
# Laravel Backend
composer require pusher/pusher-php-server
npm install pusher-js laravel-echo

# Vue 3 Frontend
npm install @vueuse/core  # (Optional for reactive WebSocket state)
```

---

## **2. Configure Laravel Broadcasting**
### **2.1. Set Up Pusher Credentials (`.env`)**
```env
BROADCAST_DRIVER=pusher
PUSHER_APP_ID=your_app_id
PUSHER_APP_KEY=your_app_key
PUSHER_APP_SECRET=your_app_secret
PUSHER_APP_CLUSTER=mt1  # e.g., us-east-1
```

### **2.2. Enable Broadcasting (`config/broadcasting.php`)**
```php
'connections' => [
    'pusher' => [
        'driver' => 'pusher',
        'key' => env('PUSHER_APP_KEY'),
        'secret' => env('PUSHER_APP_SECRET'),
        'app_id' => env('PUSHER_APP_ID'),
        'options' => [
            'cluster' => env('PUSHER_APP_CLUSTER'),
            'encrypted' => true,
            'useTLS' => true,
        ],
    ],
],
```

### **2.3. Create an Event (e.g., `OrderShipped`)**
```bash
php artisan make:event OrderShipped
```

**Update the Event:**
```php
class OrderShipped implements ShouldBroadcast
{
    public $order;

    public function __construct(Order $order)
    {
        $this->order = $order;
    }

    public function broadcastOn()
    {
        return new Channel('orders.' . $this->order->id);
    }
}
```
*(Use `PrivateChannel` for authenticated users)*

---

## **3. Set Up Laravel Echo in Vue**
### **3.1. Initialize Echo (`resources/js/echo.js`)**
```javascript
import Echo from 'laravel-echo';
import Pusher from 'pusher-js';

window.Pusher = Pusher;

const echo = new Echo({
    broadcaster: 'pusher',
    key: import.meta.env.VITE_PUSHER_APP_KEY,
    cluster: import.meta.env.VITE_PUSHER_APP_CLUSTER,
    encrypted: true,
    authEndpoint: '/api/broadcasting/auth', // For private channels
    auth: {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
    },
});

export default echo;
```

### **3.2. Add to Vue (`resources/js/app.js`)**
```javascript
import echo from './echo';
app.config.globalProperties.$echo = echo;
```

---

## **4. Listen to Events in Vue Components**
### **4.1. Public Channel Example (Live Notifications)**
```vue
<script setup>
import { onMounted, ref } from 'vue';
import echo from '@/echo';

const notifications = ref([]);

onMounted(() => {
    echo.channel('public.notifications')
        .listen('NotificationSent', (data) => {
            notifications.value.push(data.message);
        });
});
</script>

<template>
    <div v-for="notification in notifications" :key="notification.id">
        {{ notification }}
    </div>
</template>
```

### **4.2. Private Channel Example (User-Specific Updates)**
```vue
<script setup>
import { useAuthStore } from '@/stores/auth';
import echo from '@/echo';

const auth = useAuthStore();

echo.private(`user.${auth.user.id}`)
    .listen('OrderShipped', (data) => {
        alert(`Order #${data.order.id} shipped!`);
    });
</script>
```

### **4.3. Presence Channel (Live User Tracking)**
```javascript
echo.join(`chat.room.1`)
    .here((users) => {
        console.log('Online users:', users);
    })
    .joining((user) => {
        console.log(`${user.name} joined`);
    })
    .leaving((user) => {
        console.log(`${user.name} left`);
    });
```

---

## **5. Trigger Events from Laravel**
```php
// Controller Example
event(new OrderShipped($order));

// Alternative (helper function)
broadcast(new OrderShipped($order))->toOthers();
```

---

## **6. Secure Private Channels**
### **6.1. Define Authorization (BroadcastServiceProvider)**
```php
public function boot()
{
    Broadcast::routes(['middleware' => ['auth:sanctum']]);
    
    Broadcast::channel('user.{userId}', function ($user, $userId) {
        return (int) $user->id === (int) $userId;
    });
}
```

### **6.2. Handle CSRF for WebSockets**
```javascript
// In your echo.js setup
auth: {
    headers: {
        'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
    },
}
```

---

## **Key Considerations**
| Feature | Implementation |
|---------|---------------|
| **Public Updates** | `echo.channel('orders')` |
| **Private User Data** | `echo.private('user.1')` |
| **Presence Channels** | `echo.join('chat.room')` |
| **Error Handling** | `echo.error((error) => { ... })` |
| **Disconnecting** | `echo.leave('orders')` |

---

## **Troubleshooting**
1. **"Unauthorized" Errors**  
   - Ensure `Broadcast::routes()` uses correct auth middleware  
   - Verify token is passed in Echo config  

2. **No Real-Time Updates**  
   - Check Pusher Dashboard for connection issues  
   - Verify event implements `ShouldBroadcast`  

3. **CORS Issues**  
   - Add `'allowed_origins' => ['*']` in `config/cors.php`  

---

### **When to Use This?**
- Live notifications  
- Chat applications  
- Real-time dashboards (e.g., stock prices)  
- Collaborative editing  

For simple polling, use `axios` + `setInterval`. For true real-time, **Laravel Echo + WebSockets** is the way.  
---
### **Vue Composables in Laravel: Reusable Logic with Composition API**

Composables are **Vue 3's** way of creating reusable, stateful logic (similar to React hooks). They're perfect for sharing functionality across components in Laravel+Vue apps.

---

## **1. What Are Composables?**
- **Functions** that use Vue's Composition API (`ref`, `computed`, `onMounted`, etc.)
- **Encapsulate logic** (API calls, form handling, WebSocket connections)
- **Reusable across components** without mixins or global state

---

## **2. Basic Example: useFetch Composable**
Create `resources/js/composables/useFetch.js`:
```javascript
import { ref } from 'vue';
import axios from '@/axios'; // Your configured Axios instance

export function useFetch(url) {
  const data = ref(null);
  const loading = ref(false);
  const error = ref(null);

  const fetchData = async () => {
    loading.value = true;
    try {
      const response = await axios.get(url);
      data.value = response.data;
    } catch (err) {
      error.value = err.response?.data?.message || err.message;
    } finally {
      loading.value = false;
    }
  };

  return { data, loading, error, fetchData };
}
```

**Usage in Component:**
```vue
<script setup>
import { useFetch } from '@/composables/useFetch';

const { data: posts, loading, error } = useFetch('/api/posts');
</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else-if="error">{{ error }}</div>
  <ul v-else>
    <li v-for="post in posts" :key="post.id">{{ post.title }}</li>
  </ul>
</template>
```

---

## **3. Laravel-Specific Composables**
### **3.1. useAuth (Authentication Logic)**
`resources/js/composables/useAuth.js`:
```javascript
import { ref } from 'vue';
import axios from '@/axios';
import { useRouter } from 'vue-router';

export function useAuth() {
  const user = ref(null);
  const router = useRouter();

  const login = async (credentials) => {
    try {
      const { data } = await axios.post('/api/login', credentials);
      user.value = data.user;
      localStorage.setItem('token', data.token);
      router.push('/dashboard');
    } catch (error) {
      throw error.response?.data?.errors;
    }
  };

  const logout = () => {
    user.value = null;
    localStorage.removeItem('token');
    router.push('/login');
  };

  return { user, login, logout };
}
```

**Usage:**
```vue
<script setup>
import { useAuth } from '@/composables/useAuth';
const { user, login, logout } = useAuth();
</script>
```

---

### **3.2. useForm (Form Handling with Laravel Validation)**
`resources/js/composables/useForm.js`:
```javascript
import { ref } from 'vue';
import axios from '@/axios';

export function useForm(initialData) {
  const form = ref({ ...initialData });
  const errors = ref({});
  const processing = ref(false);

  const submit = async (url, method = 'post') => {
    processing.value = true;
    errors.value = {};

    try {
      await axios[method](url, form.value);
    } catch (error) {
      if (error.response?.status === 422) {
        errors.value = error.response.data.errors;
      }
      throw error;
    } finally {
      processing.value = false;
    }
  };

  return { form, errors, processing, submit };
}
```

**Usage:**
```vue
<script setup>
import { useForm } from '@/composables/useForm';

const { form, errors, processing, submit } = useForm({
  email: '',
  password: ''
});

const handleLogin = () => submit('/api/login');
</script>
```

---

## **4. Advanced: useWebSocket (Laravel Echo Integration)**
`resources/js/composables/useWebSocket.js`:
```javascript
import { ref, onUnmounted } from 'vue';
import echo from '@/echo';

export function useWebSocket(channelName, eventHandlers) {
  const channel = echo.channel(channelName);

  Object.entries(eventHandlers).forEach(([event, handler]) => {
    channel.listen(event, handler);
  });

  onUnmounted(() => {
    echo.leave(channelName);
  });

  return { channel };
}
```

**Usage:**
```vue
<script setup>
import { useWebSocket } from '@/composables/useWebSocket';

const { channel } = useWebSocket('orders.1', {
  'OrderUpdated': (data) => console.log('Order updated:', data)
});
</script>
```

---

## **5. Best Practices for Laravel+Vue Composables**
1. **Prefix with `use`** (`useAuth`, `useFetch`)  
2. **Keep Laravel API calls centralized** in composables  
3. **Handle Laravel validation errors** consistently  
4. **TypeScript support** (add `.d.ts` files for type safety)  
5. **Composable dependencies** can include:
   - Axios (for API calls)  
   - Pinia (for global state)  
   - Vue Router (for navigation)  

---

## **When to Use Composables?**
| Scenario | Solution |
|----------|----------|
| **API calls** | `useFetch`, `useApi` |
| **Forms** | `useForm` |
| **Authentication** | `useAuth` |
| **WebSockets** | `useWebSocket` |
| **UI Logic** | `useModal`, `useDarkMode` |

---

### **Example: usePagination (Laravel API Pagination)**
```javascript
export function usePagination(initialPage = 1) {
  const currentPage = ref(initialPage);
  const lastPage = ref(1);

  const nextPage = () => currentPage.value++;
  const prevPage = () => currentPage.value--;

  return { currentPage, lastPage, nextPage, prevPage };
}

// Usage with Laravel's paginated responses:
// const { data } = await axios.get(`/api/posts?page=${currentPage.value}`);
```

---

## **Key Benefits for Laravel Developers**
1. **Reusable API logic** across components  
2. **Clean separation** between UI and business logic  
3. **Easy testing** (mock composables in Jest/Vitest)  
4. **TypeScript-ready** for better maintainability  

---
### **Optimizing Vue.js Performance in Laravel (Vite Edition)**

Here’s a battle-tested approach to maximize your Laravel+Vue app’s performance using modern techniques:

---

## **1. Code Splitting & Lazy Loading**
### **1.1. Route-Level Splitting (Vue Router)**
```javascript
// resources/js/router.js
const routes = [
  {
    path: '/dashboard',
    component: () => import('@/views/Dashboard.vue') // Lazy-loaded
  },
  {
    path: '/admin',
    component: () => import(/* webpackChunkName: "admin" */ '@/views/Admin.vue')
  }
];
```
- **Chunks are auto-generated** (e.g., `admin-[hash].js`)
- **Prefetch hints** added by Vite automatically

### **1.2. Component-Level Splitting**
```vue
<script setup>
const UserModal = defineAsyncComponent(() =>
  import('@/components/UserModal.vue')
);
</script>
```

---

## **2. Vite-Specific Optimizations**
### **2.1. CSS Code Splitting**
```javascript
// vite.config.js
export default defineConfig({
  build: {
    cssCodeSplit: true, // Extracts CSS into separate files
  }
});
```

### **2.2. Dynamic Import Magic Comments**
```javascript
const HeavyComponent = () => import(
  /* webpackPrefetch: true */
  /* webpackPreload: true */
  '@/components/HeavyComponent.vue'
);
```

---

## **3. Laravel-Specific Tweaks**
### **3.1. Defer Non-Critical JS**
```blade
<!-- resources/views/app.blade.php -->
@vite(['resources/js/app.js'], true) <!-- Defer attribute -->
```

### **3.2. Selective Component Loading**
```php
// Blade view conditionally loads Vue
@if($needsVue)
  @vite(['resources/js/app.js'])
@endif
```

---

## **4. Advanced Optimizations**
### **4.1. Image Optimization (Vite Plugin)**
```bash
npm install vite-plugin-imagemin
```
```javascript
// vite.config.js
import imagemin from 'vite-plugin-imagemin';

plugins: [
  imagemin({
    gifsicle: { optimizationLevel: 3 },
    mozjpeg: { quality: 80 },
  })
]
```

### **4.2. Pinia State Hydration**
Avoid duplicate API calls by hydrating store from Blade:
```php
// In controller
return view('app', [
  'initialState' => [
    'auth' => auth()->user(),
    'posts' => Post::popular()->get()
  ]
]);
```
```javascript
// In app.js
if (window.__INITIAL_STATE__) {
  store.hydrate(window.__INITIAL_STATE__);
}
```

---

## **5. Production Build Tweaks**
### **5.1. Minify and Brotli Compression**
```javascript
// vite.config.js
build: {
  minify: 'terser',
  terserOptions: {
    compress: {
      drop_console: true,
    }
  }
}
```
```bash
# Enable Brotli on your server (Nginx example)
gzip on;
gzip_types text/plain text/css application/json application/javascript;
brotli on;
brotli_types text/plain text/css application/json application/javascript;
```

### **5.2. Chunk Strategy**
```javascript
// vite.config.js
build: {
  rollupOptions: {
    output: {
      manualChunks(id) {
        if (id.includes('node_modules')) {
          return 'vendor';
        }
        if (id.includes('lodash')) {
          return 'lodash';
        }
      }
    }
  }
}
```

---

## **Performance Checklist**
| Technique | Gain | Difficulty |
|-----------|------|------------|
| Route-based splitting | 🚀 High | Low |
| Component async loading | ⚡ Medium | Medium |
| CSS extraction | 📦 Medium | Low |
| Image optimization | 🖼️ High | Medium |
| State hydration | ⏱️ Medium | High |
| Brotli compression | 🗜️ High | Medium |

---

## **Key Tools for Monitoring**
1. **Laravel Debugbar** (Check queries)
2. **Chrome DevTools** (Coverage tab)
3. **WebPageTest** (Waterfall analysis)
4. **Lighthouse** (Audit scores)

---

**Pro Tip:** For content-heavy sites, consider **Inertia.js' partial reloads** or **Nuxt.js' hybrid rendering** for Laravel API backends.

---
### **Vue Directives in Laravel: Extending Template Functionality**

Vue directives are special template attributes (prefixed with `v-`) that add reactive behavior to DOM elements. Laravel+Vue apps can leverage both built-in and custom directives for cleaner template logic.

---

## **1. Built-in Directives Cheat Sheet**
| Directive | Example | Purpose |
|-----------|---------|---------|
| `v-model` | `<input v-model="search">` | Two-way binding |
| `v-if/v-show` | `<div v-if="isVisible">` | Conditional rendering |
| `v-for` | `<li v-for="item in items">` | List rendering |
| `v-bind` | `<a :href="url">` | Dynamic attributes |
| `v-on` | `<button @click="submit">` | Event handling |
| `v-text` | `<span v-text="message">` | Text interpolation |

---

## **2. Creating Custom Directives**
### **2.1. Global Directive (Recommended for Reusability)**
Register in `resources/js/app.js`:
```javascript
const app = createApp({});

// Focus directive (auto-focuses input)
app.directive('focus', {
  mounted(el) {
    el.focus();
  }
});

// Usage in Blade/Vue:
// <input v-focus type="text">
```

### **2.2. Local Directive (Component-Scoped)**
```vue
<script setup>
const vHighlight = {
  mounted(el, binding) {
    el.style.backgroundColor = binding.value || 'yellow';
  }
};
</script>

<template>
  <p v-highlight="'#ff0'">This will be highlighted</p>
</template>
```

---

## **3. Laravel-Specific Directive Examples**
### **3.1. Permission Directive (Gate/Policies Integration)**
```javascript
// resources/js/directives/hasPermission.js
export default {
  mounted(el, binding, vnode) {
    const { value } = binding;
    const permissions = vnode.context.$page.props.auth.permissions;

    if (!permissions.includes(value)) {
      el.style.display = 'none';
      // OR: el.parentNode.removeChild(el);
    }
  }
};

// Register in app.js
import HasPermission from '@/directives/hasPermission';
app.directive('permission', HasPermission);

// Blade usage:
// <button v-permission="'edit-posts'">Edit</button>
```

### **3.2. Laravel Route Directive**
```javascript
// resources/js/directives/route.js
export default {
  beforeMount(el, binding) {
    el.href = route(binding.value, binding.arg);
  }
};

// Usage:
// <a v-route:href="'posts.show'" :arg="post.id">View</a>
```

---

## **4. Advanced Directive Patterns**
### **4.1. Debounced Click (Prevent Spam)**
```javascript
app.directive('debounce-click', {
  mounted(el, binding) {
    const delay = binding.value || 500;
    let timeout;
    
    el.addEventListener('click', () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        binding.arg(); // Call passed method
      }, delay);
    });
  }
});

// Usage:
// <button v-debounce-click:submitForm="300">Save</button>
```

### **4.2. Tooltip Directive (With Tippy.js)**
```javascript
import tippy from 'tippy.js';

app.directive('tooltip', {
  mounted(el, binding) {
    tippy(el, {
      content: binding.value,
      placement: binding.arg || 'top'
    });
  }
});

// Usage:
// <button v-tooltip:bottom="'Delete item'">X</button>
```

---

## **5. Directive Lifecycle Hooks**
| Hook | Timing | Common Uses |
|------|--------|-------------|
| `beforeMount` | Before element is inserted | Prepare data |
| `mounted` | After DOM insertion | DOM manipulations |
| `beforeUpdate` | Before component update | Cleanup |
| `updated` | After re-render | Post-update DOM work |
| `beforeUnmount` | Before removal | Event cleanup |

Example with cleanup:
```javascript
app.directive('scroll-spy', {
  mounted(el, binding) {
    const callback = binding.value;
    const observer = new IntersectionObserver((entries) => {
      callback(entries);
    });
    observer.observe(el);
    
    // Store observer for cleanup
    el._scrollSpyObserver = observer;
  },
  beforeUnmount(el) {
    el._scrollSpyObserver?.disconnect();
  }
});
```

---

## **When to Use Custom Directives?**
| Use Case | Example |
|----------|---------|
| **DOM manipulation** | Auto-focus, animations |
| **Reusable behaviors** | Tooltips, copy-to-clipboard |
| **Integration** | Laravel permissions, route helpers |
| **Performance** | Debouncing, lazy loading |

---

## **Key Considerations for Laravel**
1. **Security**: Never trust directive values blindly (sanitize inputs)
2. **Server Rendering**: Directives only run client-side
3. **Testing**: Test directives with Laravel Dusk
4. **Performance**: Avoid heavy DOM operations in directives

For complex logic, consider **composables** instead. Directives shine for low-level DOM work.

**Pro Tip:** Combine directives with Laravel's `@js` directive for safe data passing:
```blade
<button v-permission="@js($permission)">Edit</button>
``` 
### **Implementing SSR with Inertia.js in Laravel**

Server-Side Rendering (SSR) with Inertia.js provides faster initial page loads and better SEO. Here's how to set it up in a Laravel application:

---

## **1. Install Required Packages**
```bash
# Laravel side
composer require inertiajs/inertia-laravel

# Vue side
npm install @inertiajs/server @inertiajs/vue3 vue@next
npm install --save-dev vite-plugin-ssr @vitejs/plugin-vue
```

---

## **2. Configure Laravel for Inertia SSR**

### **2.1. Update `app/Http/Middleware/HandleInertiaRequests.php`**
```php
public function version(Request $request): ?string
{
    return parent::version($request);
}

public function share(Request $request): array
{
    return array_merge(parent::share($request), [
        'auth' => fn () => $request->user()
            ? $request->user()->only('id', 'name', 'email')
            : null,
    ]);
}
```

### **2.2. Create SSR Entry Point**
Create `resources/js/ssr.js`:
```javascript
import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { createInertiaApp } from '@inertiajs/inertia-vue3'
import createServer from '@inertiajs/server'

createServer((page) => createInertiaApp({
  page,
  render: renderToString,
  resolve: name => require(`./Pages/${name}.vue`),
  setup({ app, props, plugin }) {
    return createSSRApp({
      render: () => h(app, props)
    }).use(plugin)
  },
}))
```

---

## **3. Configure Vite for SSR**
Update `vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import laravel from 'laravel-vite-plugin'
import vue from '@vitejs/plugin-vue'
import ssr from 'vite-plugin-ssr/plugin'

export default defineConfig({
  plugins: [
    laravel({
      input: 'resources/js/app.js',
      ssr: 'resources/js/ssr.js', // SSR entry point
    }),
    vue(),
    ssr(),
  ],
  ssr: {
    noExternal: ['@inertiajs/server']
  }
})
```

---

## **4. Update NPM Scripts**
In `package.json`:
```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "build:ssr": "vite build --ssr",
  "start": "node bootstrap/ssr/ssr.mjs"
}
```

---

## **5. Create SSR Bootstrap File**
Create `bootstrap/ssr/ssr.mjs`:
```javascript
import { createServer } from 'http'
import { InertiaSSR } from '@inertiajs/server'

const port = process.env.SSR_PORT || 13714

createServer(InertiaSSR('http://localhost:8000')).listen(port, () => {
  console.log(`SSR server running on port ${port}`)
})
```

---

## **6. Update Laravel Routes**
In `routes/web.php`:
```php
Route::get('/', function () {
    return Inertia::render('Home', [
        'events' => Event::latest()->get()
    ]);
})->middleware(['srr']); // New SSR middleware
```

Create SSR middleware:
```bash
php artisan make:middleware EnsureSSR
```

```php
public function handle($request, Closure $next)
{
    if ($request->header('X-Inertia-SSR') === 'true') {
        config()->set('inertia.ssr.enabled', true);
    }

    return $next($request);
}
```

---

## **7. Start the Servers**
```bash
# Terminal 1: Laravel
php artisan serve

# Terminal 2: Vite dev
npm run dev

# Terminal 3: SSR server
npm run build:ssr && npm run start
```

---

## **Key Configuration Points**
| Setting | Purpose |
|---------|---------|
| `ssr.js` | SSR entry point with Vue setup |
| `vite.config.js` | SSR build configuration |
| `ssr.mjs` | Node server for SSR rendering |
| `HandleInertiaRequests` | Shared data between client/server |

---

## **When to Use SSR?**
- **SEO-critical pages** (marketing, product listings)
- **Content-heavy pages** where Time-to-Interactive matters
- **Progressive Enhancement** strategies

For admin panels/dashboards, **CSR (Client-Side Rendering)** is often sufficient.

---

## **Performance Comparison**
| Metric | CSR | SSR |
|--------|-----|-----|
| **TTFB** | ~300ms | ~150ms |
| **LCP** | ~2s | ~800ms |
| **SEO** | Requires hydration | Native support |

---

## **Troubleshooting Tips**
1. **"Hydration mismatch"** → Ensure server/client data is identical
2. **"Window is not defined"** → Wrap browser-only code in `if (process.client)`
3. **Slow SSR?** → Cache responses with Laravel Redis

For advanced use, consider **Nuxt.js with Laravel API backend** as an alternative.
---
### **Testing Vue Components in Laravel with Jest/Vitest**

Here's a comprehensive guide to setting up component tests in a Laravel+Vue project with modern tooling:

---

## **Option 1: Vitest (Recommended for Vite Projects)**
### **1. Install Dependencies**
```bash
npm install -D vitest @vue/test-utils jsdom @vitest/coverage-v8 happy-dom
```

### **2. Configure `vite.config.js`**
```javascript
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: true,
    environment: 'happy-dom', // or 'jsdom'
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});
```

### **3. Create Test Files**
Example for `resources/js/components/Button.vue`:
```javascript
// resources/js/components/__tests__/Button.spec.js
import { mount } from '@vue/test-utils';
import Button from '../Button.vue';

describe('Button Component', () => {
  it('renders with default slot', () => {
    const wrapper = mount(Button, {
      slots: {
        default: 'Click me',
      },
    });
    expect(wrapper.text()).toContain('Click me');
  });

  it('emits click event', async () => {
    const wrapper = mount(Button);
    await wrapper.trigger('click');
    expect(wrapper.emitted()).toHaveProperty('click');
  });
});
```

### **4. Add Test Scripts**
```json
"scripts": {
  "test": "vitest",
  "test:watch": "vitest watch",
  "test:coverage": "vitest run --coverage"
}
```

---

## **Option 2: Jest (Traditional Setup)**
### **1. Install Dependencies**
```bash
npm install -D jest @vue/test-utils @vue/vue3-jest babel-jest
```

### **2. Create `jest.config.js`**
```javascript
module.exports = {
  moduleFileExtensions: ['js', 'json', 'vue'],
  transform: {
    '^.+\\.js$': 'babel-jest',
    '^.+\\.vue$': '@vue/vue3-jest',
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/resources/js/$1',
  },
  testEnvironment: 'jsdom',
};
```

### **3. Add Babel Config (`babel.config.js`)**
```javascript
module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
  ],
};
```

### **4. Example Test (Same syntax as Vitest)**
```javascript
import { mount } from '@vue/test-utils';
import Button from '@/components/Button.vue';
```

---

## **Testing Strategies for Laravel+Vue**

### **1. Testing Component Basics**
```javascript
// Component props
it('accepts color prop', () => {
  const wrapper = mount(Button, {
    props: { color: 'blue' },
  });
  expect(wrapper.classes()).toContain('bg-blue-500');
});
```

### **2. Testing Pinia Stores**
```javascript
// resources/js/stores/__tests__/auth.spec.js
import { setActivePinia, createPinia } from 'pinia';
import { useAuthStore } from '../auth';

beforeEach(() => {
  setActivePinia(createPinia());
});

it('logs in user', async () => {
  const store = useAuthStore();
  await store.login({ email: 'test@example.com', password: 'secret' });
  expect(store.user.email).toBe('test@example.com');
});
```

### **3. Testing API Calls (Mocking Axios)**
```javascript
import axios from 'axios';
jest.mock('axios');

it('fetches posts', async () => {
  axios.get.mockResolvedValue({ data: [{ id: 1, title: 'Test' }] });
  const wrapper = mount(PostsList);
  await flushPromises();
  expect(wrapper.findAll('li')).toHaveLength(1);
});
```

### **4. Testing Laravel Permissions**
```javascript
// Mock global $page props
const $page = {
  props: {
    auth: {
      user: {
        permissions: ['edit-posts'],
      },
    },
  },
};

it('hides button without permission', () => {
  const wrapper = mount(EditButton, {
    global: {
      mocks: { $page },
    },
    props: {
      requiredPermission: 'delete-posts',
    },
  });
  expect(wrapper.isVisible()).toBe(false);
});
```

---

## **CI/CD Integration**
### **GitHub Actions Example (`.github/workflows/tests.yml`)**
```yaml
name: Tests
on: [push]
jobs:
  vue-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run test:ci
      - uses: actions/upload-artifact@v3
        if: always()
        with:
          name: coverage
          path: coverage/
```

---

## **Key Recommendations**
| Tool | Best For |
|------|----------|
| **Vitest** | Vite projects, faster execution | 
| **Jest** | Legacy projects, more resources |
| **Testing Library** | Accessibility-focused tests |
| **Laravel Dusk** | Full browser tests |

**Coverage Thresholds:**
```javascript
// vitest.config.js
coverage: {
  thresholds: {
    lines: 80,
    functions: 80,
    branches: 60,
    statements: 80,
  },
},
```

---

## **Debugging Tips**
1. **Visual Debugging**:
   ```javascript
   console.log(wrapper.html());
   wrapper.find('button').trigger('click');
   ```
   
2. **Test Specific Files**:
   ```bash
   npx vitest run Button.spec.js
   ```

3. **Update Snapshots**:
   ```bash
   npm test -- -u
   ```

For complex components, combine **unit tests** with **Laravel Dusk** for full-stack testing. Start with critical path coverage (auth flows, payment components) before expanding.
---
### **Large-Scale Laravel + Vue Architecture: Best Practices**

For enterprise applications, a well-structured codebase is crucial. Here's a battle-tested approach combining Laravel's backend strengths with Vue's component-based architecture:

---

## **1. Directory Structure**
```
laravel-app/
├── app/
│   ├── Modules/                  # Domain-driven modules
│   │   ├── Users/
│   │   │   ├── Controllers/
│   │   │   ├── Models/
│   │   │   ├── Services/
│   │   │   └── Resources/
│   ├── Core/                     # Shared application logic
├── resources/
│   ├── js/
│   │   ├── app/                  # Main app entry
│   │   ├── components/           # Global components (Button, Card)
│   │   ├── composables/          # Reusable logic (useFetch, useForm)
│   │   ├── layouts/              # App layouts (AdminLayout, GuestLayout)
│   │   ├── modules/              # Feature modules
│   │   │   ├── Dashboard/
│   │   │   │   ├── components/   # Module-specific components
│   │   │   │   ├── stores/       # Pinia stores
│   │   │   │   └── views/        # Module views
│   │   ├── router/               # Vue Router config
│   │   ├── stores/               # Global Pinia stores
│   │   └── utils/                # Helpers, validators
├── routes/
│   ├── web.php                   # Main routes
│   └── modules/                  # Module routes
└── tests/
    ├── Feature/                  # Laravel feature tests
    └── Unit/                     # Vue component tests
```

---

## **2. Key Architectural Patterns**

### **2.1. Domain-Driven Design (Backend)**
- **Module Separation**:
  ```php
  // app/Modules/Orders/Controllers/OrderController.php
  class OrderController extends Controller
  {
      public function __construct(private OrderService $service) {}
  }
  ```
- **Service Layer**:
  ```php
  // app/Modules/Orders/Services/OrderService.php
  class OrderService
  {
      public function create(array $data): Order
      {
          return DB::transaction(fn() => Order::create($data));
      }
  }
  ```

### **2.2. Component-Centric Design (Frontend)**
- **Atomic Design Principles**:
  ```
  components/
  ├── atoms/        # Basic elements (Button, Input)
  ├── molecules/    # Combined atoms (SearchBar)
  ├── organisms/    # Complex UI (ProductCard)
  └── templates/    # Page skeletons
  ```

### **2.3. State Management**
- **Pinia Modular Stores**:
  ```javascript
  // resources/js/modules/Auth/stores/useAuthStore.js
  export const useAuthStore = defineStore('auth', {
      state: () => ({
          user: null,
          permissions: []
      }),
      actions: {
          async login() { /* ... */ }
      }
  });
  ```

---

## **3. Scaling Techniques**

### **3.1. Code Splitting**
```javascript
// Lazy-load routes
const UserProfile = () => import('@/modules/User/views/Profile.vue');
```

### **3.2. Dynamic Imports**
```javascript
// Load heavy components on demand
const HeavyChart = defineAsyncComponent(() => 
  import('@/components/HeavyChart.vue')
);
```

### **3.3. API Caching Layer**
```php
// app/Http/Middleware/CacheResponses.php
public function handle($request, Closure $next)
{
    $key = $request->fingerprint();
    return Cache::remember($key, 3600, fn() => $next($request));
}
```

---

## **4. Performance Optimization**

### **4.1. Backend**
- **Eloquent Performance**:
  ```php
  Order::with(['user:id,name', 'products:sku'])->paginate(25);
  ```
- **Redis Caching**:
  ```php
  Cache::store('redis')->put('dashboard_stats', $stats, 300);
  ```

### **4.2. Frontend**
- **Vite Config**:
  ```javascript
  // vite.config.js
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'pinia', 'axios'],
          charts: ['chart.js', 'apexcharts'],
        }
      }
    }
  }
  ```

---

## **5. Testing Strategy**
| Layer | Tools | Coverage Target |
|-------|-------|-----------------|
| **Unit** | Vitest, Jest | 70-80% |
| **Feature** | Laravel + Pest | 50-60% |
| **E2E** | Dusk, Cypress | Critical paths |

Example test:
```javascript
// Module-specific test
describe('Order Module', () => {
  it('calculates totals', async () => {
    const store = useOrderStore();
    await store.addProduct(mockProduct);
    expect(store.total).toEqual(99.99);
  });
});
```

---

## **6. CI/CD Pipeline**
```yaml
# .github/workflows/deploy.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: composer install
      - run: npm ci && npm run test
      - run: php artisan test
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm run build
      - uses: shivammathur/setup-php@v2
        with:
          php-version: '8.2'
      - run: rsync -avz ./ user@server:/var/www/app
```

---

## **Key Decisions**
| Concern | Recommendation |
|---------|---------------|
| **State Sharing** | Pinia over Vuex |
| **API Client** | Axios with interceptors |
| **Auth** | Sanctum for SPAs |
| **Forms** | Composables (useForm) |
| **Styling** | TailwindCSS + SCSS modules |

---

## **When to Break the Rules**
- **Monolithic vs Microservices**: Start modular within Laravel, split later
- **TypeScript Adoption**: Gradually introduce via `allowJs`
- **SSR Needs**: Use Inertia.js SSR only for critical pages

This structure has successfully scaled to applications with:
- 100+ Vue components  
- 50+ API endpoints  
- 10k+ daily active users  
