# **How to Perform CRUD Operations in CodeIgniter?**  

CRUD (**Create, Read, Update, Delete**) operations are the foundation of any web application. CodeIgniter simplifies database interactions using **models, controllers, and views**. Let's build a CRUD system step by step using **MySQL**.

---

## **1. Set Up the Database**  

Create a table named `users` in your MySQL database:  

```sql
CREATE DATABASE ci_crud;

USE ci_crud;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
✅ **This table will store user records.**  

---

## **2. Configure Database in CodeIgniter**  

📁 Open `app/Config/Database.php` and set up your database connection:  

```php
public $default = [
    'DSN'      => '',
    'hostname' => 'localhost',
    'username' => 'root',
    'password' => '',
    'database' => 'ci_crud',
    'DBDriver' => 'MySQLi',
    'DBPrefix' => '',
    'pConnect' => false,
    'DBDebug'  => (ENVIRONMENT !== 'production'),
];
```
✅ **This connects CodeIgniter to MySQL.**  

---

## **3. Create a User Model**  

📁 `app/Models/UserModel.php`  

```php
<?php
namespace App\Models;

use CodeIgniter\Model;

class UserModel extends Model
{
    protected $table = 'users';
    protected $primaryKey = 'id';
    protected $allowedFields = ['name', 'email'];
}
```
✅ **This model allows interaction with the `users` table.**  

---

## **4. Create a Controller for CRUD Operations**  

📁 `app/Controllers/UserController.php`  

```php
<?php
namespace App\Controllers;

use App\Models\UserModel;
use CodeIgniter\Controller;

class UserController extends Controller
{
    public function index()
    {
        $model = new UserModel();
        $data['users'] = $model->findAll(); // Fetch all users
        return view('user_list', $data);
    }

    public function create()
    {
        return view('user_form');
    }

    public function store()
    {
        $model = new UserModel();

        $model->insert([
            'name'  => $this->request->getPost('name'),
            'email' => $this->request->getPost('email'),
        ]);

        return redirect()->to('/users');
    }

    public function edit($id)
    {
        $model = new UserModel();
        $data['user'] = $model->find($id);
        return view('user_edit', $data);
    }

    public function update($id)
    {
        $model = new UserModel();

        $model->update($id, [
            'name'  => $this->request->getPost('name'),
            'email' => $this->request->getPost('email'),
        ]);

        return redirect()->to('/users');
    }

    public function delete($id)
    {
        $model = new UserModel();
        $model->delete($id);
        return redirect()->to('/users');
    }
}
```
✅ **This controller manages user creation, reading, updating, and deleting.**  

---

## **5. Create Views for CRUD Operations**  

### **User List View**  

📁 `app/Views/user_list.php`  

```php
<!DOCTYPE html>
<html>
<head>
    <title>User List</title>
</head>
<body>
    <h2>User List</h2>
    <a href="<?= base_url('/users/create') ?>">Add New User</a>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Email</th>
            <th>Actions</th>
        </tr>
        <?php foreach ($users as $user): ?>
            <tr>
                <td><?= $user['id'] ?></td>
                <td><?= $user['name'] ?></td>
                <td><?= $user['email'] ?></td>
                <td>
                    <a href="<?= base_url('/users/edit/' . $user['id']) ?>">Edit</a>
                    <a href="<?= base_url('/users/delete/' . $user['id']) ?>" onclick="return confirm('Delete user?');">Delete</a>
                </td>
            </tr>
        <?php endforeach; ?>
    </table>
</body>
</html>
```

---

### **Create User Form**  

📁 `app/Views/user_form.php`  

```php
<!DOCTYPE html>
<html>
<head>
    <title>Add User</title>
</head>
<body>
    <h2>Add User</h2>
    <form action="<?= base_url('/users/store') ?>" method="post">
        <label>Name:</label>
        <input type="text" name="name" required>
        <label>Email:</label>
        <input type="email" name="email" required>
        <button type="submit">Save</button>
    </form>
</body>
</html>
```

---

### **Edit User Form**  

📁 `app/Views/user_edit.php`  

```php
<!DOCTYPE html>
<html>
<head>
    <title>Edit User</title>
</head>
<body>
    <h2>Edit User</h2>
    <form action="<?= base_url('/users/update/' . $user['id']) ?>" method="post">
        <label>Name:</label>
        <input type="text" name="name" value="<?= $user['name'] ?>" required>
        <label>Email:</label>
        <input type="email" name="email" value="<?= $user['email'] ?>" required>
        <button type="submit">Update</button>
    </form>
</body>
</html>
```

---

## **6. Define Routes for CRUD Operations**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/users', 'UserController::index');
$routes->get('/users/create', 'UserController::create');
$routes->post('/users/store', 'UserController::store');
$routes->get('/users/edit/(:num)', 'UserController::edit/$1');
$routes->post('/users/update/(:num)', 'UserController::update/$1');
$routes->get('/users/delete/(:num)', 'UserController::delete/$1');
```
✅ **These routes map to the CRUD functions in `UserController`.**  

---

## **7. Testing CRUD Functionality**  

1. Start the **CodeIgniter server**:  
   ```sh
   php spark serve
   ```
2. Open your browser and navigate to:  
   ```
   http://localhost:8080/users
   ```
3. Test **Create, Read, Update, and Delete** features.  

---

## **8. Next Steps**  

🚀 **Want to add Flash Messages?** I can help!  
🚀 **Want to integrate AJAX-based CRUD?** Let’s do it! 😊