To implement **server-side pagination** in a Laravel project using JavaScript and Blade templates, you need to fetch paginated data from the server via AJAX and dynamically update the table. Here’s a step-by-step guide to achieving this:

### **Step 1: Blade Template Setup**

You will create a Blade template to display the paginated data and implement pagination controls (next, previous, and page numbers).

```html
<!-- resources/views/data-table.blade.php -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Server-Side Pagination</title>
    <link rel="stylesheet" href="{{ asset('css/app.css') }}">
</head>
<body>
    <div class="container mt-5">
        <h2>Server-Side Pagination with JavaScript</h2>

        <!-- Table -->
        <table id="dataTable" class="table table-striped">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                </tr>
            </thead>
            <tbody>
                <!-- Dynamic rows will be inserted by JavaScript -->
            </tbody>
        </table>

        <!-- Pagination Controls -->
        <div id="paginationControls">
            <!-- Pagination will be injected here by JavaScript -->
        </div>
    </div>

    <script src="{{ asset('js/app.js') }}"></script>
    <script>
        let currentPage = 1; // Current page
        let totalPages = 1;  // Total number of pages

        // Fetch and populate data with pagination
        function fetchData(page = 1) {
            currentPage = page;
            fetch(`{{ route('data.fetch') }}?page=${page}`)
                .then(response => response.json())
                .then(data => {
                    totalPages = data.last_page;  // Set total number of pages
                    renderTable(data.data);  // Render table rows
                    renderPaginationControls(data.current_page, data.last_page);  // Render pagination controls
                })
                .catch(error => console.error("Error fetching data:", error));
        }

        // Function to render the table rows
        function renderTable(tableData) {
            const tableBody = document.querySelector("#dataTable tbody");
            tableBody.innerHTML = ""; // Clear existing rows

            tableData.forEach(item => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.name}</td>
                    <td>${item.email}</td>
                `;
                tableBody.appendChild(row);
            });
        }

        // Function to render pagination controls
        function renderPaginationControls(currentPage, totalPages) {
            const paginationControls = document.getElementById("paginationControls");
            paginationControls.innerHTML = ""; // Clear existing controls

            let paginationHtml = '';

            // Previous Page Button
            if (currentPage > 1) {
                paginationHtml += `<button onclick="fetchData(${currentPage - 1})">Previous</button>`;
            }

            // Page Number Buttons
            for (let i = 1; i <= totalPages; i++) {
                paginationHtml += `
                    <button onclick="fetchData(${i})" ${i === currentPage ? 'disabled' : ''}>${i}</button>
                `;
            }

            // Next Page Button
            if (currentPage < totalPages) {
                paginationHtml += `<button onclick="fetchData(${currentPage + 1})">Next</button>`;
            }

            paginationControls.innerHTML = paginationHtml;
        }

        // Fetch data for the first page on page load
        document.addEventListener("DOMContentLoaded", function() {
            fetchData(1);
        });
    </script>
</body>
</html>
```

### **Explanation:**

1. **Dynamic Table Rows**: The `renderTable()` function dynamically generates the table rows based on the data returned by the server.
2. **Pagination Controls**: The `renderPaginationControls()` function creates page number buttons and next/previous buttons. It enables the user to navigate between pages.
3. **Fetching Paginated Data**: The `fetchData()` function is used to fetch data from the server with the appropriate page number as a query parameter (`page=1`, `page=2`, etc.).
4. **Initial Data Load**: On page load, the first page of data is fetched and displayed using the `fetchData(1)` call inside the `DOMContentLoaded` event listener.

---

### **Step 2: Controller Method for Server-Side Pagination**

The backend Laravel controller should handle the pagination and return data in a paginated format.

#### **Controller**

```php
// app/Http/Controllers/DataController.php

namespace App\Http\Controllers;

use App\Models\YourModel;  // Use the model for your data
use Illuminate\Http\Request;

class DataController extends Controller
{
    public function fetchData(Request $request)
    {
        // Get the current page from the request or set to 1 if not present
        $page = $request->get('page', 1);
        
        // Fetch paginated data from the database
        $data = YourModel::paginate(10);  // 10 items per page, you can adjust this number
        
        // Return the paginated data as JSON
        return response()->json($data);
    }
}
```

#### **Explanation:**

1. **Paginate Data**: The `paginate()` method on the Eloquent model is used to fetch a specific number of records per page. In this example, we are fetching 10 records per page.
2. **Return JSON**: The paginated data is returned as a JSON response, which includes properties like `data` (the actual records), `current_page`, `last_page`, and `total`.

---

### **Step 3: Route Setup**

Define the route in your `routes/web.php` file to handle the request for fetching data.

```php
// routes/web.php

use App\Http\Controllers\DataController;

Route::get('/fetch-data', [DataController::class, 'fetchData'])->name('data.fetch');
```

---

### **Step 4: Testing and Adjustments**

1. **Test Pagination**:
   - Load the page and verify that the table is populated with data.
   - Check the pagination controls (previous, next, and page numbers) to navigate between pages.
   
2. **Adjust Items per Page**:
   - If you need a different number of items per page, adjust the value in the `paginate()` method (`YourModel::paginate(10)`).

3. **Handling Large Data Sets**:
   - Pagination is ideal for handling large datasets. The server only loads a subset of records at a time, reducing the strain on the client-side and improving performance.

4. **Optimize Queries**:
   - For better performance, consider using `select()` to fetch only the necessary columns from the database (e.g., `YourModel::select('id', 'name', 'email')->paginate(10)`).

---

### **Conclusion**

This solution demonstrates how to implement **server-side pagination** in a Laravel project using JavaScript and Blade templates. By using Laravel's built-in pagination functionality, you can efficiently manage large datasets while maintaining a smooth user experience. The data is fetched asynchronously using AJAX, and the table and pagination controls are dynamically updated without reloading the page.