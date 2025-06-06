### Notes on Advanced SQL Techniques: Subqueries and Common Table Expressions (CTEs)

Advanced SQL techniques enhance the flexibility and efficiency of queries. Below are detailed explanations and examples of subqueries and Common Table Expressions (CTEs).

---

#### 1. **Subqueries**

A subquery is a query nested within another SQL query. Subqueries can be used in various places, such as in the SELECT, FROM, or WHERE clauses.

##### **Types of Subqueries:**

- **Single-Row Subquery**: Returns a single row and can be used with comparison operators (`=`, `>`, `<`, etc.).
  
  **Example:**

  ```sql
  SELECT name, salary
  FROM employees
  WHERE salary > (SELECT AVG(salary) FROM employees);
  ```

- **Multiple-Row Subquery**: Returns multiple rows and can be used with `IN`, `ANY`, or `ALL`.

  **Example:**

  ```sql
  SELECT name
  FROM employees
  WHERE department_id IN (SELECT id FROM departments WHERE location = 'New York');
  ```

- **Correlated Subquery**: A subquery that refers to columns from the outer query. It executes once for each row processed by the outer query.

  **Example:**

  ```sql
  SELECT e1.name
  FROM employees e1
  WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e1.department_id = e2.department_id);
  ```

---

#### 2. **Common Table Expressions (CTEs)**

A Common Table Expression (CTE) provides a temporary result set that can be referenced within a SELECT, INSERT, UPDATE, or DELETE statement. CTEs can improve the readability and structure of complex queries.

##### **Basic Syntax:**

```sql
WITH cte_name AS (
    SELECT columns
    FROM table
    WHERE condition
)
SELECT columns
FROM cte_name
WHERE another_condition;
```

##### **Example of a CTE:**

```sql
WITH department_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT e.name, e.salary, d.avg_salary
FROM employees e
JOIN department_avg d ON e.department_id = d.department_id
WHERE e.salary > d.avg_salary;
```

##### **Recursive CTEs:**

Recursive CTEs are useful for dealing with hierarchical data, such as organizational structures or tree-like data.

**Example of a Recursive CTE:**

```sql
WITH RECURSIVE employee_hierarchy AS (
    SELECT id, name, manager_id
    FROM employees
    WHERE manager_id IS NULL  -- Start with top-level managers
    UNION ALL
    SELECT e.id, e.name, e.manager_id
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy;
```

---

#### 3. **Best Practices**

- **Use CTEs for Clarity**: CTEs can make complex queries easier to read and maintain by breaking down the logic into smaller, manageable parts.

- **Optimize Subqueries**: When using subqueries, ensure they are necessary and consider using JOINs instead if performance is an issue, as they might execute multiple times.

- **Test Performance**: Always analyze the performance of queries, especially when using subqueries and CTEs, to ensure they don't negatively impact the database performance.

---

### Conclusion

Advanced SQL techniques such as subqueries and Common Table Expressions (CTEs) provide powerful tools for querying and manipulating data in a more organized and efficient manner. Understanding and applying these techniques can significantly enhance your SQL skills and improve the performance of your database interactions.