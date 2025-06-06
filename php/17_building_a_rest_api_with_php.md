Building a **REST API in PHP** involves setting up routing, handling requests, implementing authentication, and formatting responses in JSON. Here's how you can do it step by step:

---

## 1. **Setting Up Routing**
Routing in a PHP REST API is responsible for mapping incoming requests (e.g., `GET /users`, `POST /users`) to specific functions.

### Basic Routing Example:
```php
$requestMethod = $_SERVER["REQUEST_METHOD"];
$requestUri = explode("/", trim($_SERVER["REQUEST_URI"], "/"));

// Example: If request is GET /users
if ($requestMethod == "GET" && $requestUri[0] == "users") {
    getUsers();
} elseif ($requestMethod == "POST" && $requestUri[0] == "users") {
    createUser();
} else {
    response(404, ["message" => "Not Found"]);
}
```

---

## 2. **Handling Requests**
You need to read input data and process it based on the request method.

### Example: Handling GET and POST requests
```php
function getUsers() {
    // Fetch users from a database (example array for simplicity)
    $users = [
        ["id" => 1, "name" => "Alice"],
        ["id" => 2, "name" => "Bob"]
    ];
    
    response(200, $users);
}

function createUser() {
    $inputData = json_decode(file_get_contents("php://input"), true);
    
    if (!isset($inputData["name"])) {
        response(400, ["error" => "Name is required"]);
    }
    
    // Here you would insert into a database
    $newUser = ["id" => rand(3, 100), "name" => $inputData["name"]];
    
    response(201, $newUser);
}
```

---

## 3. **Implementing Authentication**
A REST API usually requires authentication, such as **JWT (JSON Web Token)** or an **API key**.

### Example: Simple Token Authentication
```php
function authenticate() {
    $headers = apache_request_headers();
    
    if (!isset($headers["Authorization"])) {
        response(401, ["error" => "Unauthorized"]);
    }

    $token = $headers["Authorization"];
    
    // Validate token (For simplicity, checking against a static token)
    if ($token !== "Bearer my_secret_token") {
        response(403, ["error" => "Forbidden"]);
    }
}
```
#### **Usage**
Call `authenticate();` at the start of protected endpoints.

---

## 4. **Formatting Responses in JSON**
A REST API should return JSON responses with appropriate HTTP status codes.

### Example: JSON Response Function
```php
function response($statusCode, $data) {
    header("Content-Type: application/json");
    http_response_code($statusCode);
    echo json_encode($data);
    exit();
}
```
#### Example Responses:
- **Success:** `response(200, ["message" => "Success"]);`
- **Not Found:** `response(404, ["error" => "Resource not found"]);`

---

## 5. **Handling Errors**
Proper error handling ensures users get clear error messages.

### Example: Global Error Handling
```php
set_exception_handler(function ($e) {
    response(500, ["error" => $e->getMessage()]);
});
```

---

## **Final Example: A Simple PHP REST API**
```php
<?php
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization");

// Basic Routing
$requestMethod = $_SERVER["REQUEST_METHOD"];
$requestUri = explode("/", trim($_SERVER["REQUEST_URI"], "/"));

function getUsers() {
    $users = [
        ["id" => 1, "name" => "Alice"],
        ["id" => 2, "name" => "Bob"]
    ];
    response(200, $users);
}

function createUser() {
    $inputData = json_decode(file_get_contents("php://input"), true);
    if (!isset($inputData["name"])) {
        response(400, ["error" => "Name is required"]);
    }
    $newUser = ["id" => rand(3, 100), "name" => $inputData["name"]];
    response(201, $newUser);
}

// Routing Logic
if ($requestMethod == "GET" && $requestUri[0] == "users") {
    getUsers();
} elseif ($requestMethod == "POST" && $requestUri[0] == "users") {
    createUser();
} else {
    response(404, ["error" => "Not Found"]);
}

// JSON Response Function
function response($statusCode, $data) {
    header("Content-Type: application/json");
    http_response_code($statusCode);
    echo json_encode($data);
    exit();
}
?>
```

---

## **Conclusion**
✅ **Routing**: Use `$_SERVER["REQUEST_METHOD"]` and `$_SERVER["REQUEST_URI"]`.  
✅ **Handling Requests**: Use `file_get_contents("php://input")` for JSON input.  
✅ **Authentication**: Validate API keys or JWT tokens.  
✅ **Response Formatting**: Always return JSON responses with correct HTTP status codes.

Want to integrate it with a database like MySQL? You can use **PDO** or **MySQLi** to fetch and store data securely. Let me know if you need that! 🚀