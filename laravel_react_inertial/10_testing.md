Here’s the **normal response** for the topic **"How to test a Laravel + React + Inertia.js app?"** along with the suggested file name at the top:

---

### **File Name**: `10_testing.md`

---

### **How to Test a Laravel + React + Inertia.js App**

#### **Goal**
Write tests for your application to ensure its functionality and reliability.

---

#### **Step 1: Write PHPUnit Tests for Laravel Backend Logic**
1. PHPUnit is Laravel's built-in testing tool for backend logic.
2. Create a test class in the `tests/Feature` directory:
   ```bash
   php artisan make:test UserTest
   ```
3. Write a test case to verify user registration:
   ```php
   namespace Tests\Feature;

   use Tests\TestCase;
   use App\Models\User;
   use Illuminate\Foundation\Testing\RefreshDatabase;

   class UserTest extends TestCase
   {
       use RefreshDatabase;

       public function test_user_registration()
       {
           $response = $this->post('/register', [
               'name' => 'John Doe',
               'email' => 'john@example.com',
               'password' => 'password',
               'password_confirmation' => 'password',
           ]);

           $response->assertRedirect('/dashboard');
           $this->assertDatabaseHas('users', ['email' => 'john@example.com']);
       }
   }
   ```
4. Run the tests:
   ```bash
   php artisan test
   ```

---

#### **Step 2: Write Jest Tests for React Components**
1. Jest is a popular testing framework for React.
2. Install Jest and React Testing Library:
   ```bash
   npm install --save-dev jest @testing-library/react @testing-library/jest-dom
   ```
3. Create a test file for a React component, e.g., `Home.test.jsx`:
   ```jsx
   import React from 'react';
   import { render, screen } from '@testing-library/react';
   import Home from './Home';

   test('renders welcome message', () => {
       render(<Home />);
       const linkElement = screen.getByText(/Welcome to Laravel + React + Inertia.js!/i);
       expect(linkElement).toBeInTheDocument();
   });
   ```
4. Add a test script to `package.json`:
   ```json
   "scripts": {
     "test": "jest"
   }
   ```
5. Run the tests:
   ```bash
   npm test
   ```

---

#### **Step 3: Use Laravel Dusk for Browser Testing**
1. Laravel Dusk is a browser automation and testing tool.
2. Install Dusk:
   ```bash
   composer require --dev laravel/dusk
   php artisan dusk:install
   ```
3. Create a Dusk test:
   ```bash
   php artisan dusk:make LoginTest
   ```
4. Write a test case to verify the login functionality:
   ```php
   namespace Tests\Browser;

   use Tests\DuskTestCase;
   use Laravel\Dusk\Browser;
   use App\Models\User;

   class LoginTest extends DuskTestCase
   {
       public function test_user_can_login()
       {
           $user = User::factory()->create();

           $this->browse(function (Browser $browser) use ($user) {
               $browser->visit('/login')
                       ->type('email', $user->email)
                       ->type('password', 'password')
                       ->press('Login')
                       ->assertPathIs('/dashboard');
           });
       }
   }
   ```
5. Run the Dusk tests:
   ```bash
   php artisan dusk
   ```

---

#### **Key Takeaways**
- Use PHPUnit to test Laravel backend logic.
- Use Jest and React Testing Library to test React components.
- Use Laravel Dusk for browser testing and automation.

---

#### **Next Steps**
- Explore how to implement real-time features with Laravel Echo and WebSockets (see `11_realtime_features.md`).
- Learn how to use Inertia.js with TypeScript (see `12_typescript_integration.md`).

---
