### Notes on Filtering Results Using the WHERE Clause in MySQL SELECT Statements

The `WHERE` clause in MySQL is a powerful tool for filtering records in `SELECT` statements based on specific conditions. This guide outlines the key aspects of using the `WHERE` clause effectively.

---

#### 1. **Basic Syntax of the WHERE Clause**

The `WHERE` clause is used to specify a condition that must be met for the rows to be included in the result set.

- **Basic Syntax**:
  ```sql
  SELECT column1, column2 FROM table_name WHERE condition;
  ```

- **Example**:
  ```sql
  SELECT * FROM employees WHERE department = 'Sales';
  ```

This retrieves all columns from the `employees` table for records where the department is 'Sales'.

---

#### 2. **Comparison Operators**

You can use various comparison operators in the `WHERE` clause to filter data:

- **Equality**: `=`
- **Not equal**: `!=` or `<>`
- **Greater than**: `>`
- **Less than**: `<`
- **Greater than or equal to**: `>=`
- **Less than or equal to**: `<=`

- **Example**:
  ```sql
  SELECT * FROM products WHERE price > 100;
  ```

This retrieves products with a price greater than 100.

---

#### 3. **Logical Operators**

You can combine multiple conditions in the `WHERE` clause using logical operators:

- **AND**: Both conditions must be true.
- **OR**: At least one of the conditions must be true.
- **NOT**: Reverses the truth value of the condition.

- **Example using AND**:
  ```sql
  SELECT * FROM orders WHERE status = 'Shipped' AND total > 50;
  ```

This retrieves orders that are 'Shipped' and have a total greater than 50.

- **Example using OR**:
  ```sql
  SELECT * FROM customers WHERE country = 'USA' OR country = 'Canada';
  ```

This retrieves customers from either the USA or Canada.

- **Example using NOT**:
  ```sql
  SELECT * FROM products WHERE NOT category = 'Electronics';
  ```

This retrieves products that are not in the 'Electronics' category.

---

#### 4. **Using IN and NOT IN**

The `IN` operator allows you to specify multiple values in a `WHERE` clause. Conversely, `NOT IN` excludes specified values.

- **Example using IN**:
  ```sql
  SELECT * FROM employees WHERE department IN ('Sales', 'Marketing');
  ```

This retrieves employees who work in either the Sales or Marketing departments.

- **Example using NOT IN**:
  ```sql
  SELECT * FROM products WHERE category NOT IN ('Clothing', 'Footwear');
  ```

This retrieves products that are not in the Clothing or Footwear categories.

---

#### 5. **Using BETWEEN**

The `BETWEEN` operator is used to filter results within a specific range, inclusive of the endpoints.

- **Basic Syntax**:
  ```sql
  SELECT * FROM table_name WHERE column_name BETWEEN value1 AND value2;
  ```

- **Example**:
  ```sql
  SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31';
  ```

This retrieves orders placed in the year 2023.

---

#### 6. **Using LIKE for Pattern Matching**

The `LIKE` operator is used for filtering results based on pattern matching, using wildcard characters:

- `%`: Represents zero or more characters.
- `_`: Represents a single character.

- **Example**:
  ```sql
  SELECT * FROM customers WHERE name LIKE 'J%';
  ```

This retrieves customers whose names start with 'J'.

- **Example with _**:
  ```sql
  SELECT * FROM products WHERE code LIKE 'A_1';
  ```

This retrieves products with codes that start with 'A', followed by any single character, and end with '1'.

---

#### 7. **Combining Conditions**

You can combine different filtering techniques in the `WHERE` clause.

- **Example**:
  ```sql
  SELECT * FROM employees 
  WHERE (department = 'Sales' OR department = 'Marketing') 
  AND hire_date >= '2022-01-01';
  ```

This retrieves employees from either the Sales or Marketing departments who were hired on or after January 1, 2022.

---

#### 8. **NULL Values Handling**

To check for `NULL` values, use the `IS NULL` or `IS NOT NULL` operators.

- **Example for NULL**:
  ```sql
  SELECT * FROM employees WHERE manager_id IS NULL;
  ```

This retrieves employees who do not have a manager assigned.

- **Example for NOT NULL**:
  ```sql
  SELECT * FROM products WHERE stock_quantity IS NOT NULL;
  ```

This retrieves products that have a stock quantity specified.

---

### Conclusion

The `WHERE` clause is essential for filtering results in MySQL `SELECT` statements. By mastering the use of comparison and logical operators, as well as techniques like `IN`, `BETWEEN`, `LIKE`, and handling `NULL` values, you can efficiently retrieve only the data that meets your specific criteria. Understanding how to effectively apply these concepts will greatly enhance your ability to query databases in MySQL.