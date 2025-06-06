Middleware functions in Express are functions that execute during the request-response cycle of an application. They have access to the request and response objects, as well as the next middleware function in the stack. Middleware can perform a variety of tasks, such as modifying request and response objects, ending the request-response cycle, and calling the next middleware function.

### Key Features of Middleware Functions

1. **Access to Request and Response Objects**: Middleware can read and modify the request and response objects.
2. **Next Function**: Middleware can pass control to the next middleware function using the `next()` function.
3. **Order of Execution**: Middleware is executed in the order they are defined in the application. This order is significant, as it affects how requests are processed.
4. **Error Handling**: Middleware can be used to handle errors in an application.

### Types of Middleware

1. **Application-level Middleware**: These middleware functions are bound to an instance of the Express application and can be used throughout the app.

2. **Router-level Middleware**: Similar to application-level middleware, but bound to an instance of `express.Router()`.

3. **Built-in Middleware**: Express has built-in middleware like `express.json()` and `express.urlencoded()` for parsing incoming request bodies.

4. **Third-party Middleware**: You can use middleware created by others, like `morgan` for logging requests and `cors` for enabling Cross-Origin Resource Sharing.

5. **Error-handling Middleware**: These are defined with four arguments (err, req, res, next) and can catch and handle errors in the application.

### How to Use Middleware in Express

Here’s how to use middleware in an Express application:

#### Step 1: Set Up a Basic Express Application

First, ensure you have a basic Express application set up. Refer to the previous setup steps if necessary.

#### Step 2: Create Middleware Functions

1. **Create a simple logging middleware function** that logs the request method and URL:

   ```javascript
   function logger(req, res, next) {
       console.log(`${req.method} ${req.url}`);
       next(); // Call the next middleware in the stack
   }
   ```

2. **Create another middleware function** to handle user authentication:

   ```javascript
   function authenticate(req, res, next) {
       const token = req.headers['authorization'];
       if (token) {
           // Simulate authentication logic
           console.log('User authenticated');
           next(); // User is authenticated, continue to the next middleware
       } else {
           res.status(401).send('Unauthorized'); // No token provided
       }
   }
   ```

#### Step 3: Use Middleware in Your Application

Now, you can use the middleware functions in your Express application. Here's an example:

```javascript
const express = require('express');
const app = express();

// Use the logger middleware for all requests
app.use(logger);

// Use the authenticate middleware for a specific route
app.get('/protected', authenticate, (req, res) => {
    res.send('This is a protected route.');
});

// Define a public route
app.get('/', (req, res) => {
    res.send('Hello, Express.js!');
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

#### Step 4: Test the Middleware

1. **Start your server** by running:

   ```bash
   node app.js
   ```

2. **Make requests to your application**:
   - Access `http://localhost:3000/` to see the public route.
   - Access `http://localhost:3000/protected` without an authorization token to see the unauthorized message.

### Conclusion

Middleware functions in Express are powerful tools that allow you to modularize your application logic. By understanding how to create and use middleware, you can enhance your Express applications with features such as logging, authentication, error handling, and more.