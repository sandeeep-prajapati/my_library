Here’s the **normal response** for the topic **"How to authenticate users in a Laravel + React + Inertia.js app?"** along with the suggested file name at the top:

---

### **File Name**: `07_authentication.md`

---

### **How to Authenticate Users in a Laravel + React + Inertia.js App**

#### **Goal**
Implement user authentication using Laravel Sanctum or Jetstream.

---

#### **Step 1: Set Up Laravel Sanctum**
1. Install Laravel Sanctum:
   ```bash
   composer require laravel/sanctum
   ```
2. Publish the Sanctum configuration file:
   ```bash
   php artisan vendor:publish --provider="Laravel\Sanctum\SanctumServiceProvider"
   ```
3. Run the Sanctum migrations:
   ```bash
   php artisan migrate
   ```
4. Add Sanctum's middleware to your `app/Http/Kernel.php`:
   ```php
   'api' => [
       \Laravel\Sanctum\Http\Middleware\EnsureFrontendRequestsAreStateful::class,
       'throttle:api',
       \Illuminate\Routing\Middleware\SubstituteBindings::class,
   ],
   ```

---

#### **Step 2: Create Login and Registration Forms in React**
1. Create a `Login.jsx` component in `resources/js/Pages/Auth`:
   ```jsx
   import React, { useState } from 'react';
   import { useForm } from '@inertiajs/react';

   export default function Login() {
     const { data, setData, post, errors } = useForm({
       email: '',
       password: '',
     });

     const handleSubmit = (e) => {
       e.preventDefault();
       post('/login');
     };

     return (
       <form onSubmit={handleSubmit}>
         <div>
           <label htmlFor="email">Email</label>
           <input
             id="email"
             type="email"
             value={data.email}
             onChange={(e) => setData('email', e.target.value)}
           />
           {errors.email && <span>{errors.email}</span>}
         </div>
         <div>
           <label htmlFor="password">Password</label>
           <input
             id="password"
             type="password"
             value={data.password}
             onChange={(e) => setData('password', e.target.value)}
           />
           {errors.password && <span>{errors.password}</span>}
         </div>
         <button type="submit">Login</button>
       </form>
     );
   }
   ```

2. Create a `Register.jsx` component in `resources/js/Pages/Auth`:
   ```jsx
   import React, { useState } from 'react';
   import { useForm } from '@inertiajs/react';

   export default function Register() {
     const { data, setData, post, errors } = useForm({
       name: '',
       email: '',
       password: '',
       password_confirmation: '',
     });

     const handleSubmit = (e) => {
       e.preventDefault();
       post('/register');
     };

     return (
       <form onSubmit={handleSubmit}>
         <div>
           <label htmlFor="name">Name</label>
           <input
             id="name"
             type="text"
             value={data.name}
             onChange={(e) => setData('name', e.target.value)}
           />
           {errors.name && <span>{errors.name}</span>}
         </div>
         <div>
           <label htmlFor="email">Email</label>
           <input
             id="email"
             type="email"
             value={data.email}
             onChange={(e) => setData('email', e.target.value)}
           />
           {errors.email && <span>{errors.email}</span>}
         </div>
         <div>
           <label htmlFor="password">Password</label>
           <input
             id="password"
             type="password"
             value={data.password}
             onChange={(e) => setData('password', e.target.value)}
           />
           {errors.password && <span>{errors.password}</span>}
         </div>
         <div>
           <label htmlFor="password_confirmation">Confirm Password</label>
           <input
             id="password_confirmation"
             type="password"
             value={data.password_confirmation}
             onChange={(e) => setData('password_confirmation', e.target.value)}
           />
         </div>
         <button type="submit">Register</button>
       </form>
     );
   }
   ```

---

#### **Step 3: Set Up Laravel Routes**
1. Open `routes/web.php` and define routes for login and registration:
   ```php
   use Illuminate\Http\Request;
   use Inertia\Inertia;

   Route::get('/login', function () {
       return Inertia::render('Auth/Login');
   });

   Route::post('/login', function (Request $request) {
       $request->validate([
           'email' => 'required|email',
           'password' => 'required',
       ]);

       if (Auth::attempt($request->only('email', 'password'))) {
           $request->session()->regenerate();
           return redirect()->intended('/dashboard');
       }

       return back()->withErrors([
           'email' => 'The provided credentials do not match our records.',
       ]);
   });

   Route::get('/register', function () {
       return Inertia::render('Auth/Register');
   });

   Route::post('/register', function (Request $request) {
       $request->validate([
           'name' => 'required|string|max:255',
           'email' => 'required|string|email|max:255|unique:users',
           'password' => 'required|string|confirmed|min:8',
       ]);

       $user = User::create([
           'name' => $request->name,
           'email' => $request->email,
           'password' => Hash::make($request->password),
       ]);

       Auth::login($user);

       return redirect('/dashboard');
   });
   ```

---

#### **Step 4: Protect Routes**
1. Use Laravel's `auth` middleware to protect routes:
   ```php
   Route::middleware('auth')->group(function () {
       Route::get('/dashboard', function () {
           return Inertia::render('Dashboard');
       });
   });
   ```

---

#### **Step 5: Run the Development Server**
1. Start the Laravel development server:
   ```bash
   php artisan serve
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Visit `http://localhost:8000/login` and `http://localhost:8000/register` to test the authentication flow.

---

#### **Key Takeaways**
- Use Laravel Sanctum for API authentication.
- Create login and registration forms in React using Inertia.js's `useForm` hook.
- Protect routes using Laravel's `auth` middleware.

---
