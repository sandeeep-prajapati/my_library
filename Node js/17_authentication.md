Implementing user authentication in an Express application typically involves several steps, including setting up user registration, login, and session management. Here’s a step-by-step guide to creating a basic authentication system using Express, bcrypt for password hashing, and express-session for session management.

### Step 1: Set Up Your Express Application

First, if you haven’t already, create a new directory for your Express application and initialize it:

```bash
mkdir auth-app
cd auth-app
npm init -y
npm install express bcrypt express-session mongoose dotenv
```

### Step 2: Set Up Mongoose and Connect to MongoDB

Create a file named `app.js` and set up Mongoose to connect to a MongoDB database. If you don't have MongoDB installed, you can use MongoDB Atlas for a cloud database.

```javascript
const express = require('express');
const mongoose = require('mongoose');
const session = require('express-session');
const bcrypt = require('bcrypt');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies
app.use(express.json()); // Parse JSON bodies
app.use(session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
}));

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.log(err));

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 3: Create a User Model

Create a directory named `models` and add a file called `User.js` for the user model:

```javascript
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true },
});

const User = mongoose.model('User', userSchema);

module.exports = User;
```

### Step 4: User Registration

Add user registration routes in your `app.js` file:

```javascript
const User = require('./models/User');

// Register Route
app.post('/register', async (req, res) => {
    const { username, password } = req.body;

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = new User({ username, password: hashedPassword });

    try {
        await newUser.save();
        res.status(201).send('User registered successfully!');
    } catch (err) {
        res.status(400).send('Error registering user: ' + err.message);
    }
});
```

### Step 5: User Login

Add login functionality to your `app.js` file:

```javascript
// Login Route
app.post('/login', async (req, res) => {
    const { username, password } = req.body;

    const user = await User.findOne({ username });
    if (!user) {
        return res.status(400).send('Invalid credentials');
    }

    // Compare password with hashed password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
        return res.status(400).send('Invalid credentials');
    }

    // Save user session
    req.session.userId = user._id;
    res.send('Logged in successfully!');
});
```

### Step 6: Protecting Routes

You can create middleware to protect routes and ensure that only authenticated users can access them:

```javascript
// Middleware to check authentication
const checkAuth = (req, res, next) => {
    if (!req.session.userId) {
        return res.status(401).send('Unauthorized');
    }
    next();
};

// Protected Route Example
app.get('/protected', checkAuth, (req, res) => {
    res.send('This is a protected route!');
});
```

### Step 7: Logout Functionality

Implement a logout route to destroy the user session:

```javascript
// Logout Route
app.post('/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).send('Error logging out');
        }
        res.send('Logged out successfully!');
    });
});
```

### Step 8: Complete Example

Here’s the complete code for your `app.js`:

```javascript
const express = require('express');
const mongoose = require('mongoose');
const session = require('express-session');
const bcrypt = require('bcrypt');
const dotenv = require('dotenv');
const User = require('./models/User'); // Import the User model

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
}));

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.log(err));

// Register Route
app.post('/register', async (req, res) => {
    const { username, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ username, password: hashedPassword });

    try {
        await newUser.save();
        res.status(201).send('User registered successfully!');
    } catch (err) {
        res.status(400).send('Error registering user: ' + err.message);
    }
});

// Login Route
app.post('/login', async (req, res) => {
    const { username, password } = req.body;
    const user = await User.findOne({ username });

    if (!user || !(await bcrypt.compare(password, user.password))) {
        return res.status(400).send('Invalid credentials');
    }

    req.session.userId = user._id;
    res.send('Logged in successfully!');
});

// Middleware to check authentication
const checkAuth = (req, res, next) => {
    if (!req.session.userId) {
        return res.status(401).send('Unauthorized');
    }
    next();
};

// Protected Route Example
app.get('/protected', checkAuth, (req, res) => {
    res.send('This is a protected route!');
});

// Logout Route
app.post('/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).send('Error logging out');
        }
        res.send('Logged out successfully!');
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 9: Testing the Application

1. **Start your Express server**:

   ```bash
   node app.js
   ```

2. **Use a tool like Postman to test the endpoints**:

   - **Register a new user**: `POST http://localhost:3000/register` with a body like:
     ```json
     {
       "username": "alice",
       "password": "password123"
     }
     ```
   - **Login**: `POST http://localhost:3000/login` with the same body.
   - **Access protected route**: `GET http://localhost:3000/protected` (make sure to be logged in).
   - **Logout**: `POST http://localhost:3000/logout`.

### Summary

- **User Registration**: Create a new user and hash the password using bcrypt.
- **User Login**: Verify credentials and manage sessions.
- **Protect Routes**: Use middleware to restrict access to certain routes.
- **Logout**: Provide functionality to destroy user sessions.

This setup gives you a basic but functional user authentication system in your Express application. You can enhance this by adding email verification, password reset functionality, and using JWT for token-based authentication.