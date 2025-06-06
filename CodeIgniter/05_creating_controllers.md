# **Creating Controllers and Handling HTTP Requests in CodeIgniter** 🚀  

Controllers in CodeIgniter are responsible for handling HTTP requests and interacting with models and views. This guide will cover:  
✅ Creating a controller  
✅ Handling HTTP GET, POST, PUT, and DELETE requests  
✅ Sending responses  

---

## **1. Creating a Controller in CodeIgniter**  

### **Step 1: Create a New Controller**  
Controllers are stored in the `app/Controllers/` directory.  

Create a new file:  
📁 `app/Controllers/UserController.php`

```php
<?php

namespace App\Controllers;

use CodeIgniter\RESTful\ResourceController;

class UserController extends ResourceController
{
    public function index()
    {
        return "Welcome to the User Controller!";
    }
}
```

📌 Now, visiting `http://localhost/my_project/usercontroller` will show:  
```
Welcome to the User Controller!
```

---

## **2. Defining Controller Methods**  

A controller method maps to a URL and handles different HTTP requests.

```php
<?php

namespace App\Controllers;

class UserController extends BaseController
{
    public function index()
    {
        return "List of users";
    }

    public function show($id)
    {
        return "User ID: " . $id;
    }

    public function create()
    {
        return "Create a new user";
    }
}
```

---

## **3. Handling HTTP Requests in CodeIgniter**  

### **a) Handling GET Requests**  

To retrieve data:  

```php
public function getUsers()
{
    return "Fetching users...";
}
```

🔗 Route in `app/Config/Routes.php`:  

```php
$routes->get('users', 'UserController::getUsers');
```

Visiting `http://localhost/my_project/users` will call `getUsers()`.

---

### **b) Handling POST Requests**  

For handling form submissions or API requests:

```php
public function createUser()
{
    $request = service('request'); 
    $name = $request->getPost('name');
    
    return "User $name has been created!";
}
```

🔗 Define POST route:  

```php
$routes->post('users/create', 'UserController::createUser');
```

To test via **cURL**:

```sh
curl -X POST -d "name=John Doe" http://localhost/my_project/users/create
```

---

### **c) Handling PUT Requests**  

For updating user data:

```php
public function updateUser($id)
{
    $request = service('request');
    $name = $request->getRawInput()['name'];

    return "User ID $id updated to $name!";
}
```

🔗 Define PUT route:

```php
$routes->put('users/update/(:num)', 'UserController::updateUser/$1');
```

Test with **cURL**:

```sh
curl -X PUT -d "name=New Name" http://localhost/my_project/users/update/2
```

---

### **d) Handling DELETE Requests**  

For deleting a user:

```php
public function deleteUser($id)
{
    return "User ID $id has been deleted!";
}
```

🔗 Define DELETE route:

```php
$routes->delete('users/delete/(:num)', 'UserController::deleteUser/$1');
```

Test with **cURL**:

```sh
curl -X DELETE http://localhost/my_project/users/delete/2
```

---

## **4. Sending JSON Responses**  

For API responses, return JSON instead of text.

```php
public function getJsonResponse()
{
    $data = [
        'status' => 'success',
        'message' => 'Data retrieved successfully',
        'users' => [
            ['id' => 1, 'name' => 'Alice'],
            ['id' => 2, 'name' => 'Bob']
        ]
    ];

    return $this->response->setJSON($data);
}
```

🔗 Route:

```php
$routes->get('users/json', 'UserController::getJsonResponse');
```

Output:

```json
{
    "status": "success",
    "message": "Data retrieved successfully",
    "users": [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
}
```

---

## **5. Redirecting Users**  

Redirect users after an action:

```php
public function redirectToHome()
{
    return redirect()->to('/home');
}
```

---

## **6. Using Middleware for Request Handling**  

You can apply middleware (filters) to handle authentication, validation, or logging.

Example: Add an **Auth Filter** in `app/Filters/Auth.php`.

---

## **Conclusion**  

✔ **Controllers** manage requests and responses  
✔ **GET, POST, PUT, DELETE** requests are handled using routes  
✔ **JSON responses** are useful for APIs  
✔ **Middleware and filters** can enhance request handling  
