Managing cookies and sessions in Express applications is essential for maintaining user state and providing a better user experience. Here’s a guide on how to implement both cookies and sessions in your Express app.

### Step 1: Set Up Your Express Application

If you haven’t already set up an Express application, start by creating a new directory and initializing a project:

```bash
mkdir cookie-session-app
cd cookie-session-app
npm init -y
npm install express cookie-parser express-session
```

### Step 2: Set Up Basic Express Server

Create a file named `app.js` and set up a basic Express server:

```javascript
const express = require('express');
const cookieParser = require('cookie-parser');
const session = require('express-session');

const app = express();
const PORT = 3000;

// Middleware
app.use(cookieParser()); // Parse cookies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies
app.use(express.json()); // Parse JSON bodies

// Set up session management
app.use(session({
    secret: 'your-secret-key', // Replace with a strong secret key
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true in production with HTTPS
}));

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 3: Working with Cookies

You can set, get, and delete cookies using the `cookie-parser` middleware. Here’s how:

#### Setting a Cookie

```javascript
app.get('/set-cookie', (req, res) => {
    res.cookie('username', 'Alice', { maxAge: 900000, httpOnly: true });
    res.send('Cookie has been set!');
});
```

#### Getting a Cookie

```javascript
app.get('/get-cookie', (req, res) => {
    const username = req.cookies.username;
    if (username) {
        res.send(`Username from cookie: ${username}`);
    } else {
        res.send('No cookie found!');
    }
});
```

#### Deleting a Cookie

```javascript
app.get('/delete-cookie', (req, res) => {
    res.clearCookie('username');
    res.send('Cookie has been deleted!');
});
```

### Step 4: Working with Sessions

Sessions allow you to store data about a user across multiple requests. Here’s how to manage sessions:

#### Setting Session Data

```javascript
app.get('/set-session', (req, res) => {
    req.session.username = 'Alice'; // Store data in session
    res.send('Session data has been set!');
});
```

#### Getting Session Data

```javascript
app.get('/get-session', (req, res) => {
    const username = req.session.username;
    if (username) {
        res.send(`Username from session: ${username}`);
    } else {
        res.send('No session data found!');
    }
});
```

#### Destroying a Session

```javascript
app.get('/destroy-session', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.send('Error destroying session!');
        }
        res.send('Session has been destroyed!');
    });
});
```

### Complete Example

Here’s the complete code for your `app.js`:

```javascript
const express = require('express');
const cookieParser = require('cookie-parser');
const session = require('express-session');

const app = express();
const PORT = 3000;

// Middleware
app.use(cookieParser());
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Set up session management
app.use(session({
    secret: 'your-secret-key', // Replace with a strong secret key
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true in production with HTTPS
}));

// Set a cookie
app.get('/set-cookie', (req, res) => {
    res.cookie('username', 'Alice', { maxAge: 900000, httpOnly: true });
    res.send('Cookie has been set!');
});

// Get a cookie
app.get('/get-cookie', (req, res) => {
    const username = req.cookies.username;
    if (username) {
        res.send(`Username from cookie: ${username}`);
    } else {
        res.send('No cookie found!');
    }
});

// Delete a cookie
app.get('/delete-cookie', (req, res) => {
    res.clearCookie('username');
    res.send('Cookie has been deleted!');
});

// Set session data
app.get('/set-session', (req, res) => {
    req.session.username = 'Alice';
    res.send('Session data has been set!');
});

// Get session data
app.get('/get-session', (req, res) => {
    const username = req.session.username;
    if (username) {
        res.send(`Username from session: ${username}`);
    } else {
        res.send('No session data found!');
    }
});

// Destroy session
app.get('/destroy-session', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.send('Error destroying session!');
        }
        res.send('Session has been destroyed!');
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 5: Testing the Application

1. **Start your Express server**:

   ```bash
   node app.js
   ```

2. **Use a browser or tool like Postman to test the endpoints**:

   - **Set a cookie**: `GET http://localhost:3000/set-cookie`
   - **Get a cookie**: `GET http://localhost:3000/get-cookie`
   - **Delete a cookie**: `GET http://localhost:3000/delete-cookie`
   - **Set a session**: `GET http://localhost:3000/set-session`
   - **Get session data**: `GET http://localhost:3000/get-session`
   - **Destroy session**: `GET http://localhost:3000/destroy-session`

### Summary

- **Cookies**: Use `res.cookie()` to set cookies and `req.cookies` to access them. You can also clear cookies using `res.clearCookie()`.
- **Sessions**: Use `req.session` to store and retrieve session data. Destroy sessions using `req.session.destroy()`.
- **Security**: In production, always use `secure: true` in session cookies and ensure your app runs over HTTPS.

This setup allows you to effectively manage user authentication, preferences, and other temporary data in your Express applications.