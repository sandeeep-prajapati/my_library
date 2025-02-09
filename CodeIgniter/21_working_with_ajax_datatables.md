## **How to Integrate DataTables with AJAX in CodeIgniter**  

**DataTables** is a powerful jQuery plugin that enhances HTML tables with features like **search, pagination, sorting, and AJAX loading**. Integrating DataTables with **AJAX in CodeIgniter** helps improve performance by fetching data dynamically without reloading the page.  

---

## **1. Install DataTables**  
Include DataTables **CSS & JS** in your view file.

### ✅ **Include in Your View (`list_users.php`)**  
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>DataTables with AJAX</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
</head>
<body>

    <table id="userTable" class="display">
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Created At</th>
            </tr>
        </thead>
    </table>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    
    <script>
        $(document).ready(function() {
            $('#userTable').DataTable({
                "processing": true,
                "serverSide": true,
                "ajax": {
                    "url": "<?= base_url('user/getUsers') ?>",
                    "type": "POST"
                },
                "columns": [
                    { "data": "id" },
                    { "data": "name" },
                    { "data": "email" },
                    { "data": "created_at" }
                ]
            });
        });
    </script>

</body>
</html>
```
✅ **Why?**  
- Enables **server-side processing** for handling large datasets efficiently.  
- Uses **AJAX** to load data dynamically.  

---

## **2. Create Controller for AJAX Data (`User.php`)**  
📁 **`app/Controllers/User.php`**  

```php
namespace App\Controllers;
use CodeIgniter\Controller;
use App\Models\UserModel;

class User extends Controller {
    public function index() {
        return view('list_users'); // Load the view
    }

    public function getUsers() {
        $request = service('request');
        $userModel = new UserModel();

        // DataTable parameters
        $searchValue = $request->getPost('search')['value']; // Search value
        $start = $request->getPost('start'); // Pagination start
        $length = $request->getPost('length'); // Pagination length

        // Fetch data with search
        $users = $userModel
            ->like('name', $searchValue)
            ->orLike('email', $searchValue)
            ->limit($length, $start)
            ->findAll();

        // Get total count
        $totalRecords = $userModel->countAll();
        $filteredRecords = count($users);

        // Response
        $response = [
            "draw" => intval($request->getPost('draw')),
            "recordsTotal" => $totalRecords,
            "recordsFiltered" => $filteredRecords,
            "data" => $users
        ];
        
        return $this->response->setJSON($response);
    }
}
```
✅ **Why?**  
- Fetches data dynamically using **server-side processing**.  
- Supports **searching, sorting, and pagination**.  

---

## **3. Create Model for Database Interaction (`UserModel.php`)**  
📁 **`app/Models/UserModel.php`**  

```php
namespace App\Models;
use CodeIgniter\Model;

class UserModel extends Model {
    protected $table = 'users';
    protected $primaryKey = 'id';
    protected $allowedFields = ['name', 'email', 'created_at'];
}
```
✅ **Why?**  
- Defines the **users** table for fetching data.  

---

## **4. Create Database Table (`users`)**  
Run this SQL in your database:

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (name, email) VALUES
('John Doe', 'john@example.com'),
('Alice Brown', 'alice@example.com'),
('Bob Smith', 'bob@example.com');
```
✅ **Why?**  
- Provides sample data for testing.  

---

## **Final Steps**  
1️⃣ **Run the application**  
```shell
php spark serve
```
2️⃣ **Visit**: `http://localhost:8080/user`  

---

## **Conclusion**  
✔ **DataTables with AJAX** improves performance by fetching **only necessary data**.  
✔ **Server-side processing** ensures faster loading for large datasets.  
✔ **Built-in features** like **searching, sorting, and pagination** enhance usability.  

🚀 **Now your CodeIgniter app loads table data dynamically with AJAX!** 🎯