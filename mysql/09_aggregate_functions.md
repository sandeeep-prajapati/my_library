### Notes on Aggregate Functions in MySQL

Aggregate functions in MySQL are used to perform calculations on a set of values and return a single value. These functions are often used with the `GROUP BY` clause to group rows that have the same values in specified columns. Here’s an overview of common aggregate functions and their usage:

---

#### 1. **COUNT()**

- **Description**: Returns the number of rows that match a specified criterion.
  
- **Basic Syntax**:
  ```sql
  SELECT COUNT(column_name) FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT COUNT(*) AS total_employees FROM employees WHERE status = 'active';
  ```
  This counts the total number of active employees in the `employees` table.

---

#### 2. **SUM()**

- **Description**: Returns the total sum of a numeric column.

- **Basic Syntax**:
  ```sql
  SELECT SUM(column_name) FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT SUM(salary) AS total_salary FROM employees WHERE department_id = 2;
  ```
  This calculates the total salary of employees in department 2.

---

#### 3. **AVG()**

- **Description**: Returns the average value of a numeric column.

- **Basic Syntax**:
  ```sql
  SELECT AVG(column_name) FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT AVG(salary) AS average_salary FROM employees;
  ```
  This retrieves the average salary of all employees.

---

#### 4. **MAX()**

- **Description**: Returns the maximum value from a set of values.

- **Basic Syntax**:
  ```sql
  SELECT MAX(column_name) FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT MAX(salary) AS highest_salary FROM employees WHERE department_id = 3;
  ```
  This finds the highest salary among employees in department 3.

---

#### 5. **MIN()**

- **Description**: Returns the minimum value from a set of values.

- **Basic Syntax**:
  ```sql
  SELECT MIN(column_name) FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT MIN(salary) AS lowest_salary FROM employees WHERE status = 'inactive';
  ```
  This retrieves the lowest salary among inactive employees.

---

### Using Aggregate Functions with GROUP BY

Aggregate functions can be combined with the `GROUP BY` clause to perform calculations on groups of rows that share a common attribute.

- **Example**:
  ```sql
  SELECT department_id, COUNT(*) AS employee_count, AVG(salary) AS average_salary 
  FROM employees 
  GROUP BY department_id;
  ```

This query counts the number of employees and calculates the average salary for each department.

---

### Combining Aggregate Functions

You can also combine multiple aggregate functions in a single query.

- **Example**:
  ```sql
  SELECT department_id, COUNT(*) AS employee_count, SUM(salary) AS total_salary, MAX(salary) AS highest_salary 
  FROM employees 
  GROUP BY department_id;
  ```

This retrieves the total number of employees, the total salary, and the highest salary for each department.

---

### Conclusion

Aggregate functions such as COUNT, SUM, AVG, MAX, and MIN are powerful tools in MySQL for summarizing and analyzing data. Mastering these functions enables you to efficiently perform calculations and derive meaningful insights from your datasets. Using them in conjunction with the `GROUP BY` clause further enhances your ability to analyze data based on various criteria.