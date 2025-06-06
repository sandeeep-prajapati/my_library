# **How to Implement User Authentication and Role-Based Access Control in CodeIgniter?**  

User authentication and role-based access control (RBAC) are essential for securing web applications. CodeIgniter provides built-in session handling, making it easy to implement authentication and restrict access based on user roles.  

---

## **1. Setting Up Database for Authentication**  

Create a `users` table with roles:  

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255),
    role ENUM('admin', 'user', 'editor') NOT NULL DEFAULT 'user'
);
```
✅ **Role-Based Access:** Each user is assigned a role (`admin`, `user`, `editor`).  

---

## **2. Creating the User Model**  

📁 `app/Models/UserModel.php`  

```php
<?php
namespace App\Models;

use CodeIgniter\Model;

class UserModel extends Model
{
    protected $table = 'users';
    protected $primaryKey = 'id';
    protected $allowedFields = ['name', 'email', 'password', 'role'];

    public function getUserByEmail($email)
    {
        return $this->where('email', $email)->first();
    }
}
```
✅ Fetches user details based on email.  

---

## **3. Creating the Authentication Controller**  

📁 `app/Controllers/AuthController.php`  

```php
<?php
namespace App\Controllers;

use App\Models\UserModel;
use CodeIgniter\Controller;

class AuthController extends Controller
{
    public function login()
    {
        return view('auth/login');
    }

    public function processLogin()
    {
        $session = session();
        $model = new UserModel();
        $email = $this->request->getVar('email');
        $password = $this->request->getVar('password');

        $user = $model->getUserByEmail($email);

        if ($user && password_verify($password, $user['password'])) {
            $session->set([
                'user_id'   => $user['id'],
                'name'      => $user['name'],
                'email'     => $user['email'],
                'role'      => $user['role'],
                'logged_in' => true
            ]);
            return redirect()->to('/dashboard');
        } else {
            $session->setFlashdata('error', 'Invalid credentials');
            return redirect()->to('/login');
        }
    }

    public function logout()
    {
        session()->destroy();
        return redirect()->to('/login');
    }
}
```
✅ **Handles user login, session management, and logout.**  

---

## **4. Creating the Login View**  

📁 `app/Views/auth/login.php`  

```html
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
</head>
<body>
    <h2>Login</h2>
    <?php if(session()->getFlashdata('error')): ?>
        <p style="color: red;"><?= session()->getFlashdata('error') ?></p>
    <?php endif; ?>

    <form action="<?= base_url('process-login') ?>" method="post">
        <input type="email" name="email" placeholder="Email" required><br>
        <input type="password" name="password" placeholder="Password" required><br>
        <button type="submit">Login</button>
    </form>
</body>
</html>
```
✅ Simple login form with error handling.  

---

## **5. Implementing Role-Based Access Control (RBAC)**  

📁 `app/Filters/AuthFilter.php`  

```php
<?php
namespace App\Filters;

use CodeIgniter\HTTP\RequestInterface;
use CodeIgniter\HTTP\ResponseInterface;
use CodeIgniter\Filters\FilterInterface;

class AuthFilter implements FilterInterface
{
    public function before(RequestInterface $request, $arguments = null)
    {
        if (!session()->get('logged_in')) {
            return redirect()->to('/login');
        }

        if ($arguments && !in_array(session()->get('role'), $arguments)) {
            return redirect()->to('/unauthorized');
        }
    }

    public function after(RequestInterface $request, ResponseInterface $response, $arguments = null)
    {
        // Nothing to do after
    }
}
```
✅ Restricts access based on user roles.  

### **Register Filter in Config**  

📁 `app/Config/Filters.php`  

```php
public $filters = [
    'auth' => ['before' => ['dashboard', 'admin/*']]
];
```
✅ Ensures authentication before accessing the dashboard and admin routes.  

---

## **6. Applying Role-Based Restrictions in Controllers**  

📁 `app/Controllers/DashboardController.php`  

```php
<?php
namespace App\Controllers;

class DashboardController extends BaseController
{
    public function __construct()
    {
        if (!session()->get('logged_in')) {
            return redirect()->to('/login');
        }

        if (session()->get('role') !== 'admin') {
            return redirect()->to('/unauthorized');
        }
    }

    public function index()
    {
        return view('dashboard');
    }
}
```
✅ Only **admin** users can access the dashboard.  

---

## **7. Protecting Routes in `Routes.php`**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/dashboard', 'DashboardController::index', ['filter' => 'auth:admin']);
$routes->get('/login', 'AuthController::login');
$routes->post('/process-login', 'AuthController::processLogin');
$routes->get('/logout', 'AuthController::logout');
```
✅ Routes are protected based on authentication and roles.  

---

## **8. Creating an Unauthorized Page**  

📁 `app/Views/errors/unauthorized.php`  

```html
<h2>Unauthorized Access</h2>
<p>You do not have permission to view this page.</p>
<a href="<?= base_url('/dashboard') ?>">Back to Dashboard</a>
```
✅ Displays an error when accessing restricted pages.  

---

## **Final Thoughts**  

✔ **Authentication:** Users log in with email and password.  
✔ **Session Management:** User info is stored in session.  
✔ **Role-Based Access Control (RBAC):** Only authorized users can access certain routes.  
✔ **Filters & Middleware:** Used to enforce access restrictions.  

🚀 **Next Step:** Do you need help adding a **registration system with password hashing**? 😊