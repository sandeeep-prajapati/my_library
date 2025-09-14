Of course. Here is a detailed, step-by-step guide on setting up a new Laravel project with Inertia.js and Vue 3.

### **01 - Setup Laravel with Inertia.js and Vue 3**

This guide will walk you through creating a new Laravel application and integrating the Inertia.js server-side and client-side adapters to use Vue 3 as your frontend.

---

#### **Step 1: Create a New Laravel Project**

```bash
composer create-project laravel/laravel inertia-vue-app
cd inertia-vue-app
```

---

#### **Step 2: Install JavaScript Dependencies**

We'll use Vite, so first, let's install the necessary NPM packages.

```bash
npm install @vitejs/plugin-vue
npm install @inertiajs/vue3
npm install vue
```

**Explanation of packages:**
*   `@vitejs/plugin-vue`: Allows Vite to process `.vue` files.
*   `@inertiajs/vue3`: The official Inertia client-side adapter for Vue 3.
*   `vue`: The Vue 3 library itself.

---

#### **Step 3: Configure Vite (`vite.config.js`)**

Replace the contents of your `vite.config.js` file with the following configuration.

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
        vue({
            template: {
                transformAssetUrls: {
                    base: null,
                    includeAbsolute: false,
                },
            },
        }),
    ],
    resolve: {
        alias: {
            '@': '/resources/js', // Optional but helpful alias
        },
    },
});
```
**Key points:**
*   The `laravel` plugin handles triggering refreshes when Blade files are changed.
*   The `vue` plugin is essential for compiling Vue components.
*   The `transformAssetUrls` option allows you to reference static assets in your Vue components without them being processed by Vite (e.g., `<img src="/images/logo.png">`).
*   The `alias` provides a shortcut to your JS directory.

---

#### **Step 4: Install and Configure Inertia Server-Side Adapter**

First, install the Inertia server-side adapter via Composer.

```bash
composer require inertiajs/inertia-laravel
```

---

#### **Step 5: Create the Root Blade Template**

Inertia requires a single root Blade template that bootstraps your JavaScript application. Create the file `resources/views/app.blade.php`.

```blade
<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title inertia>{{ config('app.name', 'Laravel') }}</title>

        <!-- Scripts -->
        @vite(['resources/css/app.css', 'resources/js/app.js'])
    </head>
    <body class="font-sans antialiased">
        <!-- The Inertia app will be mounted here -->
        @inertia
    </body>
</html>
```
**Key points:**
*   `@vite`: Laravel's directive to load the compiled CSS and JS via Vite.
*   `@inertia`: This is the magic directive. It creates a `<div>` with the necessary `id` and `data-page` attributes containing the initial page data passed from your Laravel controller.

---

#### **Step 6: Set Up the Laravel App Handler**

You need to tell Laravel to use the Inertia middleware. Publish the Inertia middleware to your project.

```bash
php artisan inertia:middleware
```

Now, register the published `HandleInertiaRequests` middleware in your `app/Http/Kernel.php` file. Add it to the `web` middleware group, typically as the last item.

**`app/Http/Kernel.php`**
```php
'web' => [
    // ... other middleware,
    \App\Http\Middleware\HandleInertiaRequests::class, // Add this line
],
```
This middleware is crucial for sharing common data (like flash messages or the authenticated user) with every Inertia response.

---

#### **Step 7: Set Up the JavaScript Entry Point (`resources/js/app.js`)**

This is where your Vue/Inertia app is initialized. Replace the entire contents of `resources/js/app.js`:

```js
import './bootstrap';
import '../css/app.css';

import { createApp, h } from 'vue';
import { createInertiaApp } from '@inertiajs/vue3';
import { resolvePageComponent } from 'laravel-vite-plugin/inertia-helpers';

createInertiaApp({
    title: (title) => `${title} - ${appName}`,
    resolve: (name) => resolvePageComponent(`./Pages/${name}.vue`, import.meta.glob('./Pages/**/*.vue')),
    setup({ el, App, props, plugin }) {
        return createApp({ render: () => h(App, props) })
            .use(plugin)
            .mount(el);
    },
    progress: {
        color: '#4B5563', // Tailwind gray-600
    },
});
```
**Explanation:**
1.  `createInertiaApp`: Initializes the Inertia app.
2.  `resolve`: A function that tells Inertia how to find your Vue page components. It uses Vite's `import.meta.glob` to lazy-load all components in the `./Pages` directory.
3.  `setup`: Creates the Vue app, uses the Inertia plugin, and mounts it to the element defined in your root template.
4.  `progress`: (Optional) Enables a navigation progress bar.

---

#### **Step 8: Create Your First Page Component**

Create the folder structure `resources/js/Pages`. Inside it, create your first page component: `Resources/js/Pages/Welcome.vue`.

```vue
<script setup>
import { Head, Link } from '@inertiajs/vue3';
</script>

<template>
  <Head title="Welcome" />

  <div class="min-h-screen bg-gray-100 flex items-center justify-center">
    <div class="text-center">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">Welcome to Inertia!</h1>
      <p class="text-lg text-gray-700 mb-6">You've successfully installed Laravel with Inertia.js and Vue 3.</p>
      <Link
        href="/about"
        class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
      >
        Go to About Page
      </Link>
    </div>
  </div>
</template>
```
**Key points:**
*   `<Head>`: An Inertia component used to set the page title and meta tags.
*   `<Link>`: An Inertia component for navigation. It's like an `<a>` tag but prevents full page reloads, making navigation incredibly fast.

---

#### **Step 9: Create a Route and Controller**

**Create a simple controller:**
```bash
php artisan make:controller PageController
```

**`app/Http/Controllers/PageController.php`**
```php
<?php

namespace App\Http\Controllers;

use Inertia\Inertia;

class PageController extends Controller
{
    public function welcome()
    {
        return Inertia::render('Welcome');
    }

    public function about()
    {
        return Inertia::render('About');
    }
}
```
**The key line is `Inertia::render('Welcome')`.** This tells Inertia to render the `Welcome.vue` component from your `js/Pages` directory instead of a Blade view.

**Define the routes in `routes/web.php`:**
```php
<?php

use App\Http\Controllers\PageController;
use Illuminate\Support\Facades\Route;

Route::get('/', [PageController::class, 'welcome']);
Route::get('/about', [PageController::class, 'about']);
```

---

#### **Step 10: Run the Development Servers**

1.  Start the Laravel development server:
    ```bash
    php artisan serve
    ```
2.  In a **separate terminal**, start the Vite development server to compile your assets:
    ```bash
    npm run dev
    ```

Now, visit [`http://localhost:8000`](http://localhost:8000) in your browser. You should see your Vue component rendered seamlessly by Laravel. Click the link to navigate to the About page (you'll need to create `About.vue`) without a full page refresh!

**To build for production:**
```bash
npm run build
```