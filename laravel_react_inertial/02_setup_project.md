Here’s the **normal response** for the topic **"How to set up a Laravel + React + Inertia.js project?"** along with the suggested file name at the top:

---

### **File Name**: `02_setup_project.md`

---

### **How to Set Up a Laravel + React + Inertia.js Project**

#### **Goal**
Create a new Laravel project and integrate React with Inertia.js.

---

#### **Step 1: Install Laravel**
1. Use Composer to create a new Laravel project:
   ```bash
   composer create-project laravel/laravel project-name
   ```
2. Navigate to the project directory:
   ```bash
   cd project-name
   ```

---

#### **Step 2: Install Inertia.js and React**
1. Install the necessary npm packages:
   ```bash
   npm install @inertiajs/react react react-dom
   ```
2. Install the Laravel Vite plugin for asset bundling:
   ```bash
   npm install -D vite laravel-vite-plugin
   ```

---

#### **Step 3: Configure `vite.config.js`**
1. Create or update the `vite.config.js` file in the project root:
   ```javascript
   import { defineConfig } from 'vite';
   import laravel from 'laravel-vite-plugin';
   import react from '@vitejs/plugin-react';

   export default defineConfig({
     plugins: [
       laravel({
         input: ['resources/css/app.css', 'resources/js/app.js'],
         refresh: true,
       }),
       react(),
     ],
   });
   ```

---

#### **Step 4: Configure `app.js` for Inertia.js**
1. Update the `resources/js/app.js` file to set up Inertia.js:
   ```javascript
   import { createInertiaApp } from '@inertiajs/react';
   import { createRoot } from 'react-dom/client';

   createInertiaApp({
     resolve: (name) => require(`./Pages/${name}`),
     setup({ el, App, props }) {
       const root = createRoot(el);
       root.render(<App {...props} />);
     },
   });
   ```

---

#### **Step 5: Set Up Laravel for Inertia.js**
1. Install the Inertia.js server-side adapter for Laravel:
   ```bash
   composer require inertiajs/inertia-laravel
   ```
2. Publish the Inertia.js middleware:
   ```bash
   php artisan inertia:middleware
   ```
3. Register the middleware in `app/Http/Kernel.php`:
   ```php
   protected $middlewareGroups = [
       'web' => [
           \App\Http\Middleware\HandleInertiaRequests::class,
           // Other middleware...
       ],
   ];
   ```

---

#### **Step 6: Create a Sample React Component**
1. Create a React component in `resources/js/Pages/Home.jsx`:
   ```jsx
   export default function Home() {
     return (
       <div>
         <h1>Welcome to Laravel + React + Inertia.js!</h1>
       </div>
     );
   }
   ```

---

#### **Step 7: Set Up a Laravel Route**
1. Update `routes/web.php` to render the React component:
   ```php
   use Inertia\Inertia;

   Route::get('/', function () {
       return Inertia::render('Home');
   });
   ```

---

#### **Step 8: Run the Development Server**
1. Start the Laravel development server:
   ```bash
   php artisan serve
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Visit `http://localhost:8000` in your browser to see the React component rendered by Laravel.

---

#### **Key Takeaways**
- Inertia.js allows Laravel to return React components directly, eliminating the need for a separate API.
- Vite is used for fast asset bundling and hot module replacement (HMR).
- The setup process involves configuring both the server-side (Laravel) and client-side (React) components.

---
