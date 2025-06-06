Implementing error handling middleware in Express applications is crucial for managing errors gracefully and ensuring that your application responds appropriately when something goes wrong. Here’s a detailed guide on how to set up error handling middleware in Express.

### Step 1: Create a Basic Express Application

First, set up a simple Express application if you haven't already:

```bash
mkdir error-handling-app
cd error-handling-app
npm init -y
npm install express
```

### Step 2: Set Up Your Express Server

Create a file named `app.js` and set up a basic Express server:

```javascript
const express = require('express');
const app = express();
const PORT = 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Sample route that may throw an error
app.get('/', (req, res) => {
    res.send('Welcome to the Error Handling App!');
});

// Route to demonstrate a synchronous error
app.get('/error', (req, res) => {
    throw new Error('This is a synchronous error!');
});

// Route to demonstrate an asynchronous error
app.get('/async-error', async (req, res, next) => {
    try {
        // Simulate an asynchronous operation that throws an error
        await Promise.reject(new Error('This is an asynchronous error!'));
    } catch (error) {
        next(error); // Pass the error to the error handling middleware
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 3: Implement Error Handling Middleware

Error handling middleware should be defined after all your routes. Here's how to implement it:

1. **Add Error Handling Middleware**:

   Add the following code after your route definitions:

   ```javascript
   // Error handling middleware
   app.use((err, req, res, next) => {
       console.error(err.stack); // Log the error for debugging

       // Send a JSON response with the error message and status code
       res.status(500).json({
           success: false,
           message: err.message || 'Internal Server Error',
       });
   });
   ```

### Complete Example

Here’s the complete code with error handling middleware:

```javascript
const express = require('express');
const app = express();
const PORT = 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Sample route that may throw an error
app.get('/', (req, res) => {
    res.send('Welcome to the Error Handling App!');
});

// Route to demonstrate a synchronous error
app.get('/error', (req, res) => {
    throw new Error('This is a synchronous error!');
});

// Route to demonstrate an asynchronous error
app.get('/async-error', async (req, res, next) => {
    try {
        await Promise.reject(new Error('This is an asynchronous error!'));
    } catch (error) {
        next(error); // Pass the error to the error handling middleware
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack); // Log the error for debugging

    // Send a JSON response with the error message and status code
    res.status(500).json({
        success: false,
        message: err.message || 'Internal Server Error',
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 4: Testing Error Handling

1. **Start your Express server**:

   ```bash
   node app.js
   ```

2. **Test the synchronous error**:

   Open your browser and navigate to `http://localhost:3000/error`. You should see a JSON response like this:

   ```json
   {
       "success": false,
       "message": "This is a synchronous error!"
   }
   ```

3. **Test the asynchronous error**:

   Navigate to `http://localhost:3000/async-error`. You should see a similar JSON response indicating that an error occurred.

### Summary

- **Error Handling Middleware**: You defined an error handling middleware function that logs errors and sends a JSON response with the error message.
- **Asynchronous Errors**: You learned how to pass errors from asynchronous code to the error handler using `next(error)`.
- **Synchronous Errors**: You handled synchronous errors thrown directly in your routes.

This setup ensures that your Express application can handle errors gracefully, providing clear feedback to clients and logging errors for debugging.