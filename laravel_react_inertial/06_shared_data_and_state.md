Here’s the **normal response** for the topic **"How to manage shared data and global state in Inertia.js?"** along with the suggested file name at the top:

---

### **File Name**: `06_shared_data_and_state.md`

---

### **How to Manage Shared Data and Global State in Inertia.js**

#### **Goal**
Share data between Laravel and React without prop drilling.

---

#### **Step 1: Use Inertia.js's `sharedProps`**
1. Inertia.js allows you to share data globally with all React components using `sharedProps`.
2. This data is automatically available in your React components as props.

---

#### **Step 2: Modify the `HandleInertiaRequests` Middleware**
1. Open the `HandleInertiaRequests` middleware (located in `app/Http/Middleware/HandleInertiaRequests.php`).
2. Add shared data in the `share` method. For example, to share the authenticated user:
   ```php
   public function share(Request $request)
   {
       return array_merge(parent::share($request), [
           'auth' => [
               'user' => $request->user() ? [
                   'id' => $request->user()->id,
                   'name' => $request->user()->name,
                   'email' => $request->user()->email,
               ] : null,
           ],
           'flash' => [
               'success' => $request->session()->get('success'),
               'error' => $request->session()->get('error'),
           ],
       ]);
   }
   ```

---

#### **Step 3: Access Shared Data in React Components**
1. The shared data is automatically passed to your React components as props.
2. For example, to access the authenticated user:
   ```jsx
   import React from 'react';

   export default function Dashboard({ auth }) {
     return (
       <div>
         <h1>Welcome, {auth.user ? auth.user.name : 'Guest'}!</h1>
       </div>
     );
   }
   ```

3. To access flash messages:
   ```jsx
   import React, { useEffect } from 'react';
   import { usePage } from '@inertiajs/react';

   export default function Home() {
     const { flash } = usePage().props;

     useEffect(() => {
       if (flash.success) {
         alert(flash.success);
       }
       if (flash.error) {
         alert(flash.error);
       }
     }, [flash]);

     return (
       <div>
         <h1>Home Page</h1>
       </div>
     );
   }
   ```

---

#### **Step 4: Run the Development Server**
1. Start the Laravel development server:
   ```bash
   php artisan serve
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Visit your application in the browser and verify that shared data (e.g., authenticated user, flash messages) is accessible in your React components.

---

#### **Key Takeaways**
- Use the `HandleInertiaRequests` middleware to share data globally with all React components.
- Shared data is automatically available in your React components as props.
- Examples of shared data include the authenticated user, flash messages, and app settings.

---

#### **Next Steps**
- Learn how to authenticate users in a Laravel + React + Inertia.js app (see `07_authentication.md`).
- Explore how to optimize performance in a Laravel + React + Inertia.js app (see `08_performance_optimization.md`).

---

Let me know if you need further assistance or the content for the next topic!