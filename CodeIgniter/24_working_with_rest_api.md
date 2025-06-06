# **How to Create RESTful APIs Using CodeIgniter?**  

Creating a RESTful API in CodeIgniter allows you to expose endpoints for **CRUD operations** that can be accessed by web and mobile applications. In this guide, you’ll learn how to build a **REST API in CodeIgniter**.  

---

## **1. Install CodeIgniter and Enable API Support**  
If you haven’t installed CodeIgniter yet, download it from [CodeIgniter’s official site](https://codeigniter.com/download) or use Composer:  
```shell
composer create-project codeigniter4/appstarter my-api
```

### **Enable CORS (Optional, for external API calls)**  
Edit **`app/Config/Filters.php`** and allow CORS:  
```php
public $globals = [
    'before' => [
        'cors' // Enable this if required
    ],
];
```

---

## **2. Configure Database and Migration**  
Update **`app/Config/Database.php`** to connect to your database.  

```php
public $default = [
    'DSN'      => '',
    'hostname' => 'localhost',
    'username' => 'root',
    'password' => '',
    'database' => 'codeigniter_api',
    'DBDriver' => 'MySQLi',
];
```

Run migration to create a users table:  
```shell
php spark migrate
```

---

## **3. Create a REST API Controller**  
📁 **`app/Controllers/UserController.php`**  
```php
namespace App\Controllers;
use CodeIgniter\RESTful\ResourceController;

class UserController extends ResourceController {

    protected $modelName = 'App\Models\UserModel';
    protected $format    = 'json';

    // GET /users
    public function index() {
        return $this->respond($this->model->findAll());
    }

    // GET /users/{id}
    public function show($id = null) {
        $user = $this->model->find($id);
        return $user ? $this->respond($user) : $this->failNotFound('User not found');
    }

    // POST /users (Create User)
    public function create() {
        $data = $this->request->getPost();
        if ($this->model->insert($data)) {
            return $this->respondCreated($data);
        }
        return $this->fail('Failed to create user');
    }

    // PUT /users/{id} (Update User)
    public function update($id = null) {
        $data = $this->request->getRawInput();
        if ($this->model->update($id, $data)) {
            return $this->respond($data);
        }
        return $this->fail('Failed to update user');
    }

    // DELETE /users/{id} (Delete User)
    public function delete($id = null) {
        if ($this->model->delete($id)) {
            return $this->respondDeleted(['id' => $id, 'message' => 'User deleted']);
        }
        return $this->fail('Failed to delete user');
    }
}
```

---

## **4. Create a User Model**  
📁 **`app/Models/UserModel.php`**  
```php
namespace App\Models;
use CodeIgniter\Model;

class UserModel extends Model {
    protected $table = 'users';
    protected $primaryKey = 'id';
    protected $allowedFields = ['name', 'email', 'password'];
}
```

---

## **5. Define API Routes**  
📁 **`app/Config/Routes.php`**  
```php
$routes->resource('users', ['controller' => 'UserController']);
```
Now, your API supports:  
✅ `GET /users` → Fetch all users  
✅ `GET /users/{id}` → Fetch a user  
✅ `POST /users` → Create a user  
✅ `PUT /users/{id}` → Update a user  
✅ `DELETE /users/{id}` → Delete a user  

---

## **6. Test the API**  
### **Using Postman or Curl**  

#### **Get all users**
```shell
curl -X GET http://localhost:8080/users
```

#### **Create a user**
```shell
curl -X POST http://localhost:8080/users -d "name=John&email=john@example.com&password=123456"
```

#### **Update a user**
```shell
curl -X PUT http://localhost:8080/users/1 -d "name=John Doe"
```

#### **Delete a user**
```shell
curl -X DELETE http://localhost:8080/users/1
```

---

## **7. Secure the API (JWT Authentication - Optional)**  
For API security, implement **JWT authentication** using `firebase/php-jwt`:  
```shell
composer require firebase/php-jwt
```

Then, create a **JWTAuthHelper.php** file and modify the `UserController` to require JWT authentication.

---

## **Conclusion**  
✔ CodeIgniter makes it **easy to build REST APIs** with `ResourceController`.  
✔ Supports **CRUD operations** (`GET`, `POST`, `PUT`, `DELETE`).  
✔ Can be secured using **JWT Authentication**.  

🚀 Now you can develop and integrate RESTful APIs in your CodeIgniter project!