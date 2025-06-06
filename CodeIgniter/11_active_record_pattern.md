## **How Does CodeIgniter’s Active Record Pattern Simplify Database Operations?**  

CodeIgniter’s **Active Record** (now called **Query Builder**) provides a structured, object-oriented way to interact with databases without writing raw SQL queries. It simplifies database operations like `SELECT`, `INSERT`, `UPDATE`, and `DELETE` by using method chaining and automatic query generation.  

---

## **1. Benefits of Active Record (Query Builder) in CodeIgniter**  

✅ **Simplifies Query Writing** – No need to write complex SQL manually.  
✅ **Prevents SQL Injection** – Automatically escapes inputs.  
✅ **Improves Code Readability** – Queries are more readable and maintainable.  
✅ **Ensures Database Compatibility** – Works across multiple database systems.  

---

## **2. Fetching Data (`SELECT` Queries)**  

### **A. Fetch All Rows**  
```php
$db = \Config\Database::connect();
$query = $db->table('users')->get();
$result = $query->getResult(); 
```
✅ Retrieves all records from the `users` table.  

### **B. Fetch Single Row by ID**  
```php
$query = $db->table('users')->where('id', 1)->get();
$user = $query->getRow();
```
✅ Fetches a single record where `id = 1`.  

### **C. Selecting Specific Columns**  
```php
$query = $db->table('users')->select('name, email')->get();
```
✅ Only returns the `name` and `email` columns.  

### **D. Using `LIKE` for Search**  
```php
$query = $db->table('users')->like('name', 'John')->get();
```
✅ Finds users whose names contain 'John'.  

### **E. Sorting Results**  
```php
$query = $db->table('users')->orderBy('created_at', 'DESC')->get();
```
✅ Orders results by newest records first.  

---

## **3. Inserting Data (`INSERT` Query)**  

```php
$data = [
    'name'  => 'John Doe',
    'email' => 'john@example.com'
];

$db->table('users')->insert($data);
```
✅ Automatically generates and executes an `INSERT` SQL statement.  

### **Batch Insert Multiple Records**  
```php
$data = [
    ['name' => 'Alice', 'email' => 'alice@example.com'],
    ['name' => 'Bob', 'email' => 'bob@example.com']
];

$db->table('users')->insertBatch($data);
```
✅ Inserts multiple rows efficiently.  

---

## **4. Updating Data (`UPDATE` Query)**  

```php
$data = ['email' => 'newemail@example.com'];
$db->table('users')->where('id', 1)->update($data);
```
✅ Updates the email for the user with `id = 1`.  

---

## **5. Deleting Data (`DELETE` Query)**  

```php
$db->table('users')->where('id', 2)->delete();
```
✅ Deletes the user with `id = 2`.  

---

## **6. Using Active Record with Models**  

Instead of writing queries in controllers, use **Models** for better structure.

### **A. Creating a Model**  
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
✅ Defines the `users` table and allowed fields.  

### **B. Using the Model in a Controller**  
📁 `app/Controllers/UserController.php`  

```php
use App\Models\UserModel;

public function index()
{
    $model = new UserModel();
    $users = $model->findAll();  // Fetch all users
    return view('users_list', ['users' => $users]);
}
```
✅ Fetches all users in a **clean and structured way**.  

---

## **7. Transactions for Multiple Queries**  

To **prevent partial updates**, use transactions:  

```php
$db->transStart();

$db->table('orders')->insert(['user_id' => 1, 'total' => 500]);
$db->table('payments')->insert(['user_id' => 1, 'amount' => 500]);

$db->transComplete();
```
✅ Ensures that either **both queries execute together** or **none at all**.  

---

## **Final Thoughts**  

✔ **Active Record (Query Builder) simplifies database operations.**  
✔ **Prevents SQL Injection automatically.**  
✔ **Improves readability and maintainability.**  
✔ **Supports transactions for atomic operations.**  

🚀 **Next Step:** Would you like a CRUD example using models and views? 😊