### Notes on Primary and Foreign Keys in MySQL

Primary and foreign keys are fundamental concepts in relational databases, used to establish relationships between tables and ensure data integrity. Understanding how to define and use these keys in MySQL is essential for effective database design. Here’s a detailed overview:

---

#### 1. **Primary Key**

A primary key is a unique identifier for a record in a table. It ensures that each record can be uniquely identified and enforces entity integrity.

- **Characteristics**:
  - **Uniqueness**: No two records can have the same primary key value.
  - **Non-null**: Primary key columns cannot contain NULL values.
  - **Single Column or Composite**: A primary key can consist of a single column or a combination of multiple columns.

**Example**: Defining a primary key on a table.

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100)
);
```

In this example, `employee_id` is defined as the primary key.

---

#### 2. **Composite Primary Key**

You can create a composite primary key by combining multiple columns.

**Example**:

```sql
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

Here, the combination of `order_id` and `product_id` serves as the primary key.

---

#### 3. **Foreign Key**

A foreign key is a field (or collection of fields) in one table that uniquely identifies a row in another table. Foreign keys establish a link between the two tables and enforce referential integrity.

- **Characteristics**:
  - **References Primary Key**: A foreign key in one table points to a primary key in another table.
  - **Allows NULLs**: Foreign key columns can contain NULL values unless specified otherwise.
  - **Cascade Options**: You can specify actions to take on updates or deletes (e.g., CASCADE, SET NULL).

**Example**: Defining a foreign key in a table.

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

In this example, `customer_id` in the `orders` table is a foreign key that references the `customer_id` in the `customers` table.

---

#### 4. **Adding Foreign Keys to Existing Tables**

You can also add foreign keys to existing tables using the `ALTER TABLE` statement.

**Example**:

```sql
ALTER TABLE orders
ADD FOREIGN KEY (customer_id) REFERENCES customers(customer_id);
```

---

#### 5. **Cascading Updates and Deletes**

When defining foreign keys, you can specify what happens to the records in the child table if the referenced records in the parent table are updated or deleted.

**Example**:

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);
```

In this case, if a record in the `customers` table is deleted, all corresponding records in the `orders` table will also be deleted. Similarly, updates to the `customer_id` will cascade to the `orders` table.

---

#### 6. **Benefits of Using Keys**

- **Data Integrity**: Ensures that relationships between tables remain consistent.
- **Performance**: Improves the efficiency of data retrieval and updates.
- **Normalization**: Helps in organizing data into related tables, reducing redundancy.

---

### Conclusion

Primary and foreign keys are vital components of relational database design in MySQL. They ensure data integrity, enforce relationships between tables, and optimize query performance. By properly defining these keys, you can create a robust and efficient database schema that maintains the relationships and integrity of your data.