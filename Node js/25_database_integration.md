Integrating databases like MongoDB and MySQL with Express applications allows you to build robust applications that can store and manage data. Here’s a step-by-step guide for integrating both databases with Express.

## Integrating MongoDB with Express

### 1. **Setup MongoDB**

You can use a local MongoDB server or a cloud service like MongoDB Atlas.

- **Local MongoDB Installation**: Follow the [MongoDB installation guide](https://docs.mongodb.com/manual/installation/).
- **MongoDB Atlas**: Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and set up a cluster.

### 2. **Install Required Packages**

In your Express project, install the `mongoose` package, which is a popular ODM (Object Data Modeling) library for MongoDB.

```bash
npm install mongoose
```

### 3. **Connect to MongoDB**

In your Express app, establish a connection to the MongoDB database using Mongoose.

```javascript
// app.js or index.js
const express = require('express');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 3000;

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/mydatabase', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.error('MongoDB connection error:', err));

// Middleware to parse JSON bodies
app.use(express.json());

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### 4. **Define a Mongoose Model**

Create a model that represents the structure of your data. For example, a simple user model:

```javascript
// models/User.js
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
});

const User = mongoose.model('User', userSchema);

module.exports = User;
```

### 5. **Create Routes to Handle Data**

Create routes for CRUD operations. For example, to create and retrieve users:

```javascript
// routes/users.js
const express = require('express');
const User = require('../models/User');

const router = express.Router();

// Create a new user
router.post('/', async (req, res) => {
  const user = new User(req.body);
  try {
    await user.save();
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Get all users
router.get('/', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;
```

### 6. **Integrate the Routes**

In your main application file, integrate the routes:

```javascript
// app.js
const userRoutes = require('./routes/users');

// Use the user routes
app.use('/api/users', userRoutes);
```

### 7. **Run the Application**

Now run your application using:

```bash
node app.js
```

You can test the API using Postman or any API client.

---

## Integrating MySQL with Express

### 1. **Setup MySQL**

You can use a local MySQL server or a cloud-based MySQL service.

- **Local MySQL Installation**: Follow the [MySQL installation guide](https://dev.mysql.com/doc/refman/8.0/en/installing.html).

### 2. **Install Required Packages**

In your Express project, install the `mysql` or `mysql2` package to connect to MySQL.

```bash
npm install mysql2
```

### 3. **Connect to MySQL**

Establish a connection to the MySQL database:

```javascript
// app.js
const express = require('express');
const mysql = require('mysql2');

const app = express();
const PORT = process.env.PORT || 3000;

// Create a connection to the database
const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: 'yourpassword',
  database: 'mydatabase'
});

// Connect to MySQL
db.connect((err) => {
  if (err) {
    console.error('Database connection failed:', err);
    return;
  }
  console.log('Connected to MySQL');
});

// Middleware to parse JSON bodies
app.use(express.json());

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### 4. **Create Routes to Handle Data**

Create routes for CRUD operations. For example, to create and retrieve users:

```javascript
// routes/users.js
const express = require('express');
const db = require('../config/db'); // Assuming you've set up your MySQL connection in a separate file

const router = express.Router();

// Create a new user
router.post('/', (req, res) => {
  const { name, email } = req.body;
  const query = 'INSERT INTO users (name, email) VALUES (?, ?)';
  
  db.execute(query, [name, email], (err, results) => {
    if (err) {
      return res.status(500).json({ message: err.message });
    }
    res.status(201).json({ id: results.insertId, name, email });
  });
});

// Get all users
router.get('/', (req, res) => {
  const query = 'SELECT * FROM users';
  
  db.query(query, (err, results) => {
    if (err) {
      return res.status(500).json({ message: err.message });
    }
    res.json(results);
  });
});

module.exports = router;
```

### 5. **Integrate the Routes**

Integrate the routes into your main application file:

```javascript
// app.js
const userRoutes = require('./routes/users');

// Use the user routes
app.use('/api/users', userRoutes);
```

### 6. **Run the Application**

Now run your application using:

```bash
node app.js
```

You can test the API using Postman or any API client.

---

### Conclusion

By following these steps, you can effectively integrate both MongoDB and MySQL with your Express applications. This allows you to perform CRUD operations and manage your data efficiently. You can choose between MongoDB and MySQL based on your specific use case, as each database has its strengths and weaknesses.