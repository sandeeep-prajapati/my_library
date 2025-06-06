To implement **sorting** and **filtering** for a Blade-rendered table in a Laravel application, you can combine both front-end (JavaScript) and back-end (Laravel) techniques. The goal is to make your table interactive by allowing users to sort the columns and filter data based on certain criteria.

### **Step 1: Blade Template Setup for Sorting and Filtering**

We’ll start by rendering the table with sorting and filtering capabilities.

```html
<!-- resources/views/data-table.blade.php -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sortable and Filterable Table</title>
    <link rel="stylesheet" href="{{ asset('css/app.css') }}">
</head>
<body>
    <div class="container mt-5">
        <h2>Sortable and Filterable Table</h2>

        <!-- Search Filter Input -->
        <input type="text" id="search" class="form-control mb-3" placeholder="Search by name" onkeyup="filterTable()">

        <!-- Table -->
        <table id="dataTable" class="table table-striped">
            <thead>
                <tr>
                    <th id="sortId" onclick="sortTable(0)">ID</th>
                    <th id="sortName" onclick="sortTable(1)">Name</th>
                    <th id="sortEmail" onclick="sortTable(2)">Email</th>
                </tr>
            </thead>
            <tbody>
                <!-- Dynamic rows will be inserted by JavaScript -->
            </tbody>
        </table>
    </div>

    <script src="{{ asset('js/app.js') }}"></script>
    <script>
        // Initial data array (you will populate this with actual data from Laravel backend)
        let data = [];
        let currentSortColumn = 0; // Default sort column (ID)
        let sortOrder = 'asc'; // Default sort order

        document.addEventListener("DOMContentLoaded", function () {
            fetchDataAndPopulateTable();
        });

        // Fetch data from the Laravel backend and populate the table
        function fetchDataAndPopulateTable() {
            fetch("{{ route('data.fetch') }}")
                .then(response => response.json())
                .then(fetchedData => {
                    data = fetchedData; // Store fetched data
                    renderTable(data);  // Render the table with the fetched data
                })
                .catch(error => {
                    console.error("Error fetching data:", error);
                });
        }

        // Function to render the table
        function renderTable(tableData) {
            const tableBody = document.querySelector("#dataTable tbody");
            tableBody.innerHTML = ""; // Clear the table body

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

        // Sorting function (by column index)
        function sortTable(columnIndex) {
            // Toggle sort order
            if (currentSortColumn === columnIndex) {
                sortOrder = (sortOrder === 'asc') ? 'desc' : 'asc';
            } else {
                currentSortColumn = columnIndex;
                sortOrder = 'asc'; // Default to ascending order when switching columns
            }

            // Sort the data based on column and order
            const sortedData = [...data].sort((a, b) => {
                const valA = Object.values(a)[columnIndex];
                const valB = Object.values(b)[columnIndex];

                if (valA < valB) return (sortOrder === 'asc') ? -1 : 1;
                if (valA > valB) return (sortOrder === 'asc') ? 1 : -1;
                return 0;
            });

            renderTable(sortedData);
        }

        // Filter the table data based on search input
        function filterTable() {
            const searchTerm = document.getElementById('search').value.toLowerCase();
            const filteredData = data.filter(item => {
                return item.name.toLowerCase().includes(searchTerm); // Filter by name
            });
            renderTable(filteredData); // Render filtered data
        }
    </script>
</body>
</html>
```

### **Explanation:**

1. **Search Filter**: 
   - The `input` field with `id="search"` allows the user to filter the table by typing in a name. The `filterTable()` function will filter the data based on the name and re-render the table.
   
2. **Sorting**:
   - The table header (`<th>`) elements have `onclick` events that trigger the `sortTable()` function, which sorts the data by the clicked column.
   - `currentSortColumn` tracks the column that is being sorted, and `sortOrder` toggles between ascending (`'asc'`) and descending (`'desc'`).

3. **Dynamic Table Rendering**:
   - The `renderTable()` function populates the table with the data received from the backend. It’s used for both initial data rendering and after sorting or filtering.

4. **Data Fetching**:
   - When the page loads, it fetches the data from Laravel’s backend (via the `/fetch-data` route) using the `fetchDataAndPopulateTable()` function.

---

### **Step 2: Controller and Route Setup**

Now, let’s update the controller to fetch the data from the database and return it as a JSON response.

#### **Controller Method**

```php
// app/Http/Controllers/DataController.php

namespace App\Http\Controllers;

use App\Models\YourModel;  // Use the model for your data
use Illuminate\Http\Request;

class DataController extends Controller
{
    public function fetchData(Request $request)
    {
        // Fetch the data from the database (use pagination or no pagination depending on the size)
        $data = YourModel::all(); // You can also use paginate() for pagination if needed
        
        return response()->json($data); // Return as JSON response
    }
}
```

#### **Route Definition**

Make sure you have the route defined to fetch the data.

```php
// routes/web.php

use App\Http\Controllers\DataController;

Route::get('/fetch-data', [DataController::class, 'fetchData'])->name('data.fetch');
```

---

### **Step 3: Customize and Test**

1. **Test Search Filter**:
   - The search filter will dynamically filter rows based on the name of each entry.
   - You can adjust the `filterTable` function to filter by other columns if needed.

2. **Test Sorting**:
   - Click on the column headers (ID, Name, or Email) to sort the data.
   - Sorting alternates between ascending and descending order each time the column is clicked.

3. **Handle Larger Datasets**:
   - For larger datasets, you might want to consider implementing **pagination** or **lazy loading** to improve performance. You can modify the backend controller to return paginated results and adjust the frontend to handle pagination.

---

### **Conclusion**

This solution demonstrates how to add **sorting** and **filtering** capabilities to a Laravel Blade-rendered table using JavaScript. By handling sorting and filtering on the front-end, you can create an interactive table that allows users to search for and organize data dynamically. You can further extend this by adding additional features such as **pagination**, **multiple filters**, and **advanced search** capabilities.