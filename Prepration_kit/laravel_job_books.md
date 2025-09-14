Here’s a clear and concise explanation of the difference between `INNER JOIN`, `LEFT JOIN`, and `RIGHT JOIN` in MySQL:

---

### 🔁 **INNER JOIN**

* **Returns**: Only the **matching rows** from both tables.
* **Use case**: You want records where there's a match in both tables.

```sql
SELECT * 
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;
```

> ✅ Result: Only employees who belong to a department.

---

### ⬅️ **LEFT JOIN** (or LEFT OUTER JOIN)

* **Returns**: **All rows** from the **left table**, and matching rows from the right table. If no match, right table columns are `NULL`.
* **Use case**: You want **all employees**, even those without a department.

```sql
SELECT * 
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;
```

> ✅ Result: Every employee, with or without a department.

---

### ➡️ **RIGHT JOIN** (or RIGHT OUTER JOIN)

* **Returns**: **All rows** from the **right table**, and matching rows from the left table. If no match, left table columns are `NULL`.
* **Use case**: You want **all departments**, even if no employee is assigned.

```sql
SELECT * 
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;
```

> ✅ Result: Every department, even if it has no employees.

---

### 📌 Summary Table

| Join Type    | Matched from Both | All from Left | All from Right |
| ------------ | ----------------- | ------------- | -------------- |
| `INNER JOIN` | ✅ Yes             | ❌ No          | ❌ No           |
| `LEFT JOIN`  | ✅ Yes             | ✅ Yes         | ❌ No           |
| `RIGHT JOIN` | ✅ Yes             | ❌ No          | ✅ Yes          |


### ✅ 1. **Using `LIMIT` with `ORDER BY` and `OFFSET`**

```sql
SELECT salary 
FROM employees 
ORDER BY salary DESC 
LIMIT 1 OFFSET 1;
```

> 🔍 This skips the highest and gets the second one directly.

---

### ✅ 2. **Using a Subquery with `MAX()`**

```sql
SELECT MAX(salary) AS second_highest_salary 
FROM employees 
WHERE salary < (
    SELECT MAX(salary) FROM employees
);
```

> 🔍 Gets the max salary *less than* the actual highest salary.

---

### ✅ 3. **Using `DENSE_RANK()` (MySQL 8+)**

```sql
SELECT salary 
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk 
    FROM employees
) ranked 
WHERE rnk = 2;
```

---

### ✅ **What does `GROUP BY` do?**

The `GROUP BY` clause groups rows that have the same values in specified columns, so aggregate functions like `COUNT()`, `SUM()`, `AVG()`, etc. can be applied **per group**.

---

### 🔢 **Example: Using `COUNT()`**

**🔍 Task**: Count the number of employees in each department.

```sql
SELECT department_id, COUNT(*) AS total_employees
FROM employees
GROUP BY department_id;
```

> ✅ Groups all employees by their department and counts them.

---

### 💰 **Example: Using `SUM()`**

**🔍 Task**: Get the total salary paid per department.

```sql
SELECT department_id, SUM(salary) AS total_salary
FROM employees
GROUP BY department_id;
```

> ✅ Groups rows by `department_id` and sums up the salaries.

---

### 🧠 Tips:

* All columns in the `SELECT` list must be either:

  * In the `GROUP BY` clause, or
  * Used inside an aggregate function.
* You can also `ORDER BY` aggregated results:

  ```sql
  ORDER BY total_salary DESC;
  ```

---
Here’s the query to update all records where the `status` is `'pending'` to `'approved'`:

```sql
UPDATE your_table_name
SET status = 'approved'
WHERE status = 'pending';
```

### 🔍 Example with table name `orders`:

```sql
UPDATE orders
SET status = 'approved'
WHERE status = 'pending';
```

---

### ✅ What it does:

* `UPDATE orders`: Target the table.
* `SET status = 'approved'`: Change the value.
* `WHERE status = 'pending'`: Only rows with `'pending'` status are affected.

> ⚠️ Always back up your data or run a `SELECT` first to confirm:

```sql
SELECT * FROM orders WHERE status = 'pending';
```
### ✅ **Using `LIMIT` and `OFFSET` to Paginate Results in MySQL**

`LIMIT` and `OFFSET` are used together to fetch a specific chunk of rows (like a page) from a result set.

---

### 🔹 **Syntax**:

```sql
SELECT column1, column2
FROM table_name
ORDER BY column_name
LIMIT limit_value OFFSET offset_value;
```

---

### 🔍 **Example: Basic Pagination**

```sql
SELECT id, name
FROM users
ORDER BY id
LIMIT 10 OFFSET 0;
```

> ✅ Fetches the **first 10 rows** (Page 1)

---

### 🔁 **Paginated Pages Example**:

| Page | Query                      |
| ---- | -------------------------- |
| 1    | `LIMIT 10 OFFSET 0`        |
| 2    | `LIMIT 10 OFFSET 10`       |
| 3    | `LIMIT 10 OFFSET 20`       |
| N    | `LIMIT 10 OFFSET (N-1)*10` |

---

### 🔧 **Formula**:

```sql
OFFSET = (PageNumber - 1) * PageSize
```

---
### ✅ **Difference Between `WHERE` and `HAVING` in MySQL**

| Feature               | `WHERE`                                                  | `HAVING`                         |
| --------------------- | -------------------------------------------------------- | -------------------------------- |
| **Used with**         | Individual rows (before grouping)                        | Groups (after aggregation)       |
| **When applied**      | Before `GROUP BY`                                        | After `GROUP BY`                 |
| **Aggregate support** | ❌ Cannot use aggregate functions like `SUM()`, `COUNT()` | ✅ Used to filter aggregated data |
| **Performance**       | Faster as it's applied earlier                           | Slower if used unnecessarily     |

---

### 🔹 **Example Using `WHERE`**

```sql
SELECT * 
FROM employees
WHERE department_id = 3;
```

> ✅ Filters **rows** where `department_id` is 3 **before grouping**.

---

### 🔹 **Example Using `HAVING`**

```sql
SELECT department_id, COUNT(*) AS total
FROM employees
GROUP BY department_id
HAVING COUNT(*) > 5;
```

> ✅ Filters **groups** (departments) where employee count is **more than 5**.

---

### 🧠 Tip:

* Use `WHERE` to filter rows **before aggregation**
* Use `HAVING` to filter groups **after aggregation**

Let me know if you want combined examples using both!
### ✅ **How to Retrieve Duplicate Rows Based on a Specific Column in MySQL**

To find **duplicate values** in a specific column, you use `GROUP BY` along with `HAVING COUNT(*) > 1`.

---

### 🔹 **Example: Find duplicate emails in a `users` table**

```sql
SELECT email, COUNT(*) AS count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

> ✅ Shows each email that appears more than once, and how many times it appears.

---

### 🔹 **If you want full row details of the duplicates:**

```sql
SELECT * 
FROM users 
WHERE email IN (
    SELECT email
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1
);
```

> ✅ Retrieves **all rows** where the email is duplicated.

---

### 🧠 Tip:

* You can replace `email` with any column (e.g., `phone`, `username`, etc.)
* Use `ORDER BY count DESC` in the subquery to see the most duplicated values.

Let me know if you also want to delete duplicates or keep just one!
### ✅ **Delete Rows Older Than 30 Days Based on `created_at`**

To delete records older than 30 days from the current date, use `NOW()` or `CURDATE()` with `INTERVAL`.

---

### 🔹 **Query:**

```sql
DELETE FROM your_table_name
WHERE created_at < NOW() - INTERVAL 30 DAY;
```

### 🔍 Example:

```sql
DELETE FROM logs
WHERE created_at < NOW() - INTERVAL 30 DAY;
```

> ✅ Deletes all rows from the `logs` table where `created_at` is **more than 30 days ago**.

---

### 🧠 Tips:

* Use `CURDATE()` instead of `NOW()` if `created_at` has only the **date** part (no time):

  ```sql
  WHERE created_at < CURDATE() - INTERVAL 30 DAY;
  ```
* Always run a `SELECT` first to preview affected rows:

  ```sql
  SELECT * FROM logs WHERE created_at < NOW() - INTERVAL 30 DAY;
  ```

### ✅ **Using Subqueries in `SELECT`, `WHERE`, and `FROM` Clauses in MySQL**

Subqueries (also called **nested queries**) allow you to use the result of one query inside another. Let's explore how to use them in each clause with examples:

---

### 🔹 1. **Subquery in `SELECT` Clause**

**Use case**: Fetch each employee along with the **average salary** of all employees.

```sql
SELECT name, salary, 
       (SELECT AVG(salary) FROM employees) AS avg_salary
FROM employees;
```

> ✅ This adds the same average salary as a new column for every row.

---

### 🔹 2. **Subquery in `WHERE` Clause**

**Use case**: Get employees whose salary is **above the average** salary.

```sql
SELECT name, salary 
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

> ✅ The subquery filters rows based on a calculated value.

---

### 🔹 3. **Subquery in `FROM` Clause**

**Use case**: First calculate average salary per department, then filter on that.

```sql
SELECT dept_avg.department_id, dept_avg.avg_salary
FROM (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
) AS dept_avg
WHERE dept_avg.avg_salary > 50000;
```

> ✅ The subquery becomes a **derived table** that you can query like a normal table.

---

### 🧠 Tips:

* Use **aliases** when subqueries are used in `FROM`.
* Subqueries can return:

  * A **single value** (scalar)
  * A **list** (for `IN`)
  * A **table** (for `FROM` clause)

### ✅ **Query to Calculate Total Sales Per Month from a `sales` Table**

Assuming the table `sales` has at least these columns:

* `sale_amount` (numeric)
* `sale_date` (DATE or DATETIME)

---

### 🔹 **Query**:

```sql
SELECT 
    DATE_FORMAT(sale_date, '%Y-%m') AS sale_month,
    SUM(sale_amount) AS total_sales
FROM sales
GROUP BY sale_month
ORDER BY sale_month;
```

---

### 🔍 What it does:

* `DATE_FORMAT(sale_date, '%Y-%m')`: Extracts the **year and month** like `2025-08`.
* `SUM(sale_amount)`: Calculates total sales for that month.
* `GROUP BY sale_month`: Groups sales by month.
* `ORDER BY sale_month`: Sorts results chronologically.

---
### ✅ **Hide All `<div>` Elements with Class `alert` Using jQuery**

To hide all `<div class="alert">` elements, you can use the `jQuery .hide()` method with a class selector:

---

### 🔹 **Code:**

```javascript
$('.alert').hide();
```

> ✅ This will instantly hide **all elements** with the class `alert` (not just `<div>`s — any element).

---

### 🔹 If you want to **specifically target `<div>`s**:

```javascript
$('div.alert').hide();
```

---

### 🔹 Optionally with animation:

```javascript
$('div.alert').fadeOut(300); // fades out in 300ms
```

---
### ✅ **Difference Between `.on()` and `.click()` in jQuery Event Handling**

Both `.on()` and `.click()` are used to attach event handlers — but they have different scopes and flexibility.

---

### 🔹 **1. `.click()`**

* Binds a **click event** **directly** to existing elements.
* Does **not** work for dynamically added elements after the page has loaded.

```javascript
$('#btn').click(function() {
  alert('Button clicked!');
});
```

> ✅ Works only if `#btn` exists at the time this code runs.

---

### 🔹 **2. `.on()`**

* More **versatile**: supports **multiple event types**, **event delegation**, and **dynamic elements**.
* Best for attaching handlers to elements that **may not exist yet** when the script runs.

```javascript
$(document).on('click', '#btn', function() {
  alert('Button clicked!');
});
```

> ✅ Works for both existing and future `#btn` elements.

---

### 🧠 Summary Table

| Feature                  | `.click()` | `.on('click', ...)` |
| ------------------------ | ---------- | ------------------- |
| Binds only `click`       | ✅ Yes      | ✅ Yes               |
| Works with dynamic DOM   | ❌ No       | ✅ Yes               |
| Supports delegation      | ❌ No       | ✅ Yes               |
| Use for best flexibility | ❌ Limited  | ✅ Recommended       |

---
### ✅ **Performing an AJAX `GET` Request and Handling the Response in jQuery**

You can use jQuery’s `$.ajax()` or shorthand `$.get()` to perform asynchronous HTTP GET requests.

---

### 🔹 **Using `$.ajax()`**

```javascript
$.ajax({
  url: '/api/data',        // URL to send request to
  type: 'GET',             // HTTP method
  success: function(response) {
    console.log('Data received:', response);  // handle response
  },
  error: function(xhr, status, error) {
    console.error('AJAX Error:', error);     // handle error
  }
});
```

---

### 🔹 **Using shorthand `$.get()`**

```javascript
$.get('/api/data', function(response) {
  console.log('Data received:', response);
});
```

---

### 🧠 Example Use Case:

```javascript
// Fetch user info and display in a div
$.get('/api/user', function(data) {
  $('#userInfo').text('Name: ' + data.name);
});
```

---

### 📌 Tips:

* Make sure your server supports CORS if calling cross-origin.
* Use `dataType: 'json'` if you expect a JSON response:

  ```javascript
  $.ajax({
    url: '/api/user',
    type: 'GET',
    dataType: 'json',
    success: function(data) { ... }
  });
  ```
### ✅ **How to Add a Class to an Element When It Is Clicked (Using jQuery)**

You can use the `.addClass()` method in jQuery inside a click event handler.

---

### 🔹 **Example:**

```javascript
$('.my-button').click(function() {
  $(this).addClass('active');
});
```

> ✅ When any element with class `.my-button` is clicked, the class `active` is added to **that specific element**.

---

### 🔹 **To add class to a different target:**

```javascript
$('.my-button').click(function() {
  $('#target-div').addClass('highlight');
});
```

> ✅ When `.my-button` is clicked, `#target-div` gets the `highlight` class.

---

### 🧠 Tip:

You can also **toggle** the class on click using:

```javascript
$(this).toggleClass('active');
```
### ✅ **How to Get and Set the Value of an Input Field Using jQuery**

---

### 🔹 **Get the value of an input field**

```javascript
let value = $('#myInput').val();
console.log(value);
```

> ✅ Retrieves the current value from the input with ID `myInput`.

---

### 🔹 **Set the value of an input field**

```javascript
$('#myInput').val('Hello, World!');
```

> ✅ Sets the input field’s value to `"Hello, World!"`.

---

### 🔍 **Full Example:**

```html
<input type="text" id="myInput" placeholder="Type something">
<button id="getBtn">Get</button>
<button id="setBtn">Set</button>

<script>
  $('#getBtn').click(function() {
    alert($('#myInput').val());
  });

  $('#setBtn').click(function() {
    $('#myInput').val('Preset value');
  });
</script>
```

---

### ✅ **Difference Between `var`, `let`, and `const` in JavaScript**

These three keywords are used to **declare variables**, but they differ in scope, hoisting, and mutability.

---

### 🔹 **1. `var`**

* **Function-scoped** (not block-scoped)
* **Hoisted** (but initialized as `undefined`)
* Can be **redeclared** and **reassigned**

```javascript
var x = 10;
if (true) {
  var x = 20;  // same variable, updated
}
console.log(x); // ➜ 20
```

---

### 🔹 **2. `let`**

* **Block-scoped**
* **Hoisted**, but **not initialized** (access before declaration causes `ReferenceError`)
* Can be **reassigned**, but **not redeclared** in the same scope

```javascript
let y = 10;
if (true) {
  let y = 20;  // new variable (block-scoped)
  console.log(y); // ➜ 20
}
console.log(y); // ➜ 10
```

---

### 🔹 **3. `const`**

* **Block-scoped**
* Must be **initialized at declaration**
* Cannot be **reassigned** or **redeclared**
* For objects and arrays, **contents can change**, but **reference can't**

```javascript
const z = 10;
// z = 20; // ❌ Error: Assignment to constant variable

const user = { name: "Sandeep" };
user.name = "Om";  // ✅ Allowed (object property changed)
```

---

### 🧠 Summary Table:

| Feature        | `var`           | `let`     | `const`   |
| -------------- | --------------- | --------- | --------- |
| Scope          | Function        | Block     | Block     |
| Hoisted        | Yes (undefined) | Yes (TDZ) | Yes (TDZ) |
| Redeclarable   | ✅ Yes           | ❌ No      | ❌ No      |
| Reassignable   | ✅ Yes           | ✅ Yes     | ❌ No      |
| Initialization | Optional        | Optional  | Required  |

---
### ✅ **Using `map()`, `filter()`, and `reduce()` on an Array in JavaScript**

These three are powerful **array methods** in JavaScript for transforming and processing data.

---

### 🔹 **1. `map()` — Transform Each Element**

> Returns a **new array** with the result of applying a function to each item.

```javascript
const numbers = [1, 2, 3];
const doubled = numbers.map(num => num * 2);
console.log(doubled); // ➜ [2, 4, 6]
```

---

### 🔹 **2. `filter()` — Select Some Elements**

> Returns a **new array** with only the elements that **pass a test**.

```javascript
const numbers = [1, 2, 3, 4];
const evens = numbers.filter(num => num % 2 === 0);
console.log(evens); // ➜ [2, 4]
```

---

### 🔹 **3. `reduce()` — Accumulate into One Value**

> Applies a function to each item, **accumulating** a single result.

```javascript
const numbers = [1, 2, 3, 4];
const total = numbers.reduce((acc, num) => acc + num, 0);
console.log(total); // ➜ 10
```

* `acc` is the **accumulator**
* `num` is the current value
* `0` is the **initial value**

---

### 🧠 Summary Table

| Method     | Purpose                | Returns      |
| ---------- | ---------------------- | ------------ |
| `map()`    | Transform each element | New array    |
| `filter()` | Select based on a test | New array    |
| `reduce()` | Combine into one value | Single value |

---
### ✅ **What is Event Delegation in JavaScript?**

**Event Delegation** is a technique where you **attach a single event listener to a parent element** instead of multiple listeners to individual child elements. It works by taking advantage of **event bubbling**.

---

### 🧠 **Why Use Event Delegation?**

* Improves **performance** (fewer listeners)
* Handles **dynamic elements** (added after DOM load)
* Keeps your code **clean and efficient**

---

### 🔹 **How It Works (Event Bubbling)**

When an event occurs on a child element, it **bubbles up** to its parent, grandparent, etc.

So you can:

1. Attach a listener to the parent
2. Catch events from children using `event.target`

---

### 🔍 **Example:**

```html
<ul id="menu">
  <li>Home</li>
  <li>About</li>
  <li>Contact</li>
</ul>
```

```javascript
document.getElementById('menu').addEventListener('click', function(e) {
  if (e.target.tagName === 'LI') {
    console.log('You clicked:', e.target.textContent);
  }
});
```

> ✅ A single listener handles clicks on **any `<li>`**, even new ones added later!

---

### 🧠 Key Points:

* Use `e.target` to find the actual clicked element.
* Useful when dealing with many elements or **dynamic content**.
* Works with any bubbling events like `click`, `input`, `submit`.

---
### ✅ **How to Create a Simple Promise in JavaScript with `.then()` and `.catch()`**

A **Promise** represents an asynchronous operation — it can be:

* **Resolved** (✅ success)
* **Rejected** (❌ failure)

---

### 🔹 **1. Creating a Promise**

```javascript
const myPromise = new Promise((resolve, reject) => {
  const success = true; // simulate success or failure

  if (success) {
    resolve('✅ Operation was successful!');
  } else {
    reject('❌ Something went wrong.');
  }
});
```

---

### 🔹 **2. Handling the Promise with `.then()` and `.catch()`**

```javascript
myPromise
  .then(result => {
    console.log(result);  // runs if resolved
  })
  .catch(error => {
    console.error(error); // runs if rejected
  });
```

---

### 🧪 **Full Working Example:**

```javascript
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const data = { name: 'Sandeep', role: 'Developer' };
      const success = true;

      if (success) resolve(data);
      else reject('Failed to fetch data');
    }, 1000);
  });
}

fetchData()
  .then(data => console.log('Received:', data))
  .catch(err => console.error('Error:', err));
```

---
### ✅ **What is Hoisting in JavaScript?**

**Hoisting** is JavaScript’s default behavior of **moving declarations to the top** of their scope (before code execution).

But there's a catch:

---

### 🔹 **1. Variable Hoisting**

* Variables declared with `**var**` are **hoisted** — but **only the declaration**, **not the assignment**.
* Variables declared with `**let**` and `**const**` are **hoisted** too, but placed in the **Temporal Dead Zone (TDZ)** — accessing them before declaration causes an error.

```javascript
console.log(x); // undefined (not ReferenceError)
var x = 5;
```

```javascript
console.log(y); // ❌ ReferenceError
let y = 10;
```

---

### 🔹 **2. Function Hoisting**

* **Function declarations** are hoisted **with their body**, so you can call them before they're defined:

```javascript
greet(); // ✅ Works
function greet() {
  console.log('Hello!');
}
```

* **Function expressions** (especially with `const` or `let`) are **not fully hoisted**:

```javascript
sayHi(); // ❌ TypeError: sayHi is not a function
const sayHi = function() {
  console.log('Hi!');
};
```

---

### 🧠 Summary Table:

| Type                 | Hoisted? | Initialized? | Access Before Declaration |
| -------------------- | -------- | ------------ | ------------------------- |
| `var`                | ✅ Yes    | ❌ No         | ✅ undefined               |
| `let` / `const`      | ✅ Yes    | ❌ No         | ❌ ReferenceError (TDZ)    |
| Function Declaration | ✅ Yes    | ✅ Yes        | ✅ Allowed                 |
| Function Expression  | ❌ No     | ❌ No         | ❌ Error                   |

---
### ✅ **Difference Between `include`, `require`, `include_once`, and `require_once` in PHP**

These functions are used to **import PHP files** into other PHP files — but they behave slightly differently.

---

### 🔹 1. `include`

* Includes the file **every time it’s called**
* **Warning** if file not found, script **continues execution**

```php
include 'header.php';
```

---

### 🔹 2. `require`

* Also includes the file every time
* **Fatal error** if file not found, script **stops execution**

```php
require 'config.php';
```

---

### 🔹 3. `include_once`

* Includes the file **only once**, even if called multiple times
* **Warning** on error, script continues

```php
include_once 'functions.php';
```

---

### 🔹 4. `require_once`

* Includes the file **only once**
* **Fatal error** on failure, script stops

```php
require_once 'db.php';
```

---

### 🧠 Summary Table:

| Keyword        | Error on Missing File | Includes Only Once | Execution Continues |
| -------------- | --------------------- | ------------------ | ------------------- |
| `include`      | ⚠️ Warning            | ❌ No               | ✅ Yes               |
| `require`      | ❌ Fatal Error         | ❌ No               | ❌ No                |
| `include_once` | ⚠️ Warning            | ✅ Yes              | ✅ Yes               |
| `require_once` | ❌ Fatal Error         | ✅ Yes              | ❌ No                |

---

### ✅ Use Tips:

* Use `require_once` for **core config or DB files** to avoid multiple inclusions.
* Use `include` for **optional files** (like banners, ads, etc.).

Let me know if you'd like a real project folder structure example using these!
### ✅ **Handling Form Data via `POST` and `GET` in PHP**

PHP provides two **superglobal arrays** to access submitted form data:

* `$_GET` — for data sent via the **GET method**
* `$_POST` — for data sent via the **POST method**

---

### 🔹 **1. HTML Form Example**

```html
<form method="POST" action="submit.php">
  <input type="text" name="username" placeholder="Enter your name">
  <input type="submit" value="Submit">
</form>
```

---

### 🔹 **2. Handling `POST` Data in `submit.php`**

```php
<?php
if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $username = $_POST['username']; // gets the form input
    echo "Hello, " . htmlspecialchars($username);
}
?>
```

> ✅ `htmlspecialchars()` prevents XSS attacks.

---

### 🔹 **3. Handling `GET` Data**

```html
<form method="GET" action="search.php">
  <input type="text" name="query" placeholder="Search...">
  <input type="submit" value="Go">
</form>
```

```php
<?php
if ($_SERVER["REQUEST_METHOD"] === "GET") {
    $query = $_GET['query'];
    echo "You searched for: " . htmlspecialchars($query);
}
?>
```

---

### 🧠 Tip:

* Always validate and sanitize user inputs.
* Use `$_REQUEST` to access both `$_GET` and `$_POST` (not recommended for security-sensitive apps).

---
### ✅ **How PHP Handles Sessions Using `session_start()` and `$_SESSION`**

Sessions in PHP allow you to **store user-specific data across multiple pages**, like login state, cart items, etc.

---

### 🔹 **1. `session_start()`**

* Initializes a **new session** or resumes an **existing one**
* Must be called **at the top** of the script (before any output)

```php
<?php
session_start();
?>
```

> ✅ Creates a session ID and stores it in a cookie (`PHPSESSID`)

---

### 🔹 **2. Storing Data in `$_SESSION`**

```php
<?php
session_start();
$_SESSION['username'] = 'Sandeep';
$_SESSION['role'] = 'admin';
?>
```

> ✅ Values are stored on the server and available across all pages using the same session.

---

### 🔹 **3. Accessing Session Data**

```php
<?php
session_start();
echo $_SESSION['username']; // ➜ Sandeep
?>
```

---

### 🔹 **4. Removing Session Data**

* **Unset a variable**:

```php
unset($_SESSION['username']);
```

* **Destroy the entire session**:

```php
session_destroy();
```

> ⚠️ After `session_destroy()`, the session is deleted but variables may still exist until the script ends.

---

### 🧠 Common Use Case: **Login System**

```php
// login.php
session_start();
$_SESSION['logged_in'] = true;
$_SESSION['user_id'] = 101;
```

```php
// dashboard.php
session_start();
if ($_SESSION['logged_in'] !== true) {
  header('Location: login.php');
  exit;
}
```

---

### 📌 Summary

| Function            | Purpose                               |
| ------------------- | ------------------------------------- |
| `session_start()`   | Starts or resumes a session           |
| `$_SESSION`         | Stores and accesses session variables |
| `unset()`           | Removes a specific session variable   |
| `session_destroy()` | Deletes the session completely        |
### ✅ **Connecting to a MySQL Database Using `mysqli` in PHP and Handling Errors**

You can use the **`mysqli`** extension in PHP to connect to a MySQL database.

---

### 🔹 **1. Basic Connection Code**

```php
<?php
$host = "localhost";
$user = "root";
$password = "";
$database = "my_database";

// Create connection
$conn = new mysqli($host, $user, $password, $database);

// Check connection
if ($conn->connect_error) {
    die("❌ Connection failed: " . $conn->connect_error);
}

echo "✅ Connected successfully";
?>
```

---

### 🔍 **Explanation:**

* `new mysqli(...)`: Opens a new connection.
* `$conn->connect_error`: Returns error message if the connection fails.
* `die()`: Stops the script if connection fails.

---

### 🔹 **2. Safer Version with Error Reporting**

```php
<?php
$mysqli = new mysqli("localhost", "root", "", "my_database");

if ($mysqli->connect_errno) {
    echo "❌ Failed to connect to MySQL: (" . $mysqli->connect_errno . ") " . $mysqli->connect_error;
    exit();
}
echo "✅ Connection OK!";
?>
```

---

### 🧠 Best Practice:

* Keep credentials in a separate `config.php` file.
* Always check for errors before running queries.
* Close the connection when done:

```php
$conn->close();
```

---
### ✅ **Associative Arrays in PHP & Looping with `foreach`**

---

### 🔹 **What is an Associative Array?**

An **associative array** in PHP uses **named keys** instead of numeric indexes.

```php
$user = [
    "name" => "Sandeep",
    "email" => "sandeep@example.com",
    "role" => "Developer"
];
```

> ✅ Keys like `"name"`, `"email"`, and `"role"` are strings, not numbers.

---

### 🔹 **Looping Through Associative Arrays Using `foreach`**

#### 🔸 Loop Through **Key + Value**:

```php
foreach ($user as $key => $value) {
    echo "$key: $value <br>";
}
```

> ✅ Output:

```
name: Sandeep
email: sandeep@example.com
role: Developer
```

---

#### 🔸 Loop Through **Values Only**:

```php
foreach ($user as $value) {
    echo "$value <br>";
}
```

> ✅ Output:

```
Sandeep
sandeep@example.com
Developer
```

---

### 🧠 Tip:

You can nest associative arrays and loop through them too:

```php
$users = [
    "u1" => ["name" => "Sandeep", "role" => "Dev"],
    "u2" => ["name" => "Om", "role" => "Designer"]
];

foreach ($users as $id => $info) {
    echo "$id: " . $info["name"] . " - " . $info["role"] . "<br>";
}
```
