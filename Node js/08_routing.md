Defining routes in Express and handling different HTTP methods is a fundamental aspect of building web applications. Express allows you to create routes that respond to various HTTP methods (GET, POST, PUT, DELETE, etc.) and to specify how the application should handle requests to those routes.

### Step-by-Step Guide to Defining Routes in Express

#### Step 1: Set Up a Basic Express Application

First, make sure you have a basic Express application set up. If you haven't done this already, follow these steps:

1. **Install Express**:

   ```bash
   npm install express
   ```

2. **Create a file named `app.js`**:

   ```javascript
   const express = require('express');
   const app = express();

   // Middleware to parse JSON request bodies
   app.use(express.json());

   const PORT = 3000;

   // Start the server
   app.listen(PORT, () => {
       console.log(`Server is running on http://localhost:${PORT}`);
   });
   ```

#### Step 2: Define Routes

You can define routes using the app methods corresponding to HTTP methods. Here’s how to handle different HTTP methods:

1. **GET Method**: Retrieve data from the server.

   ```javascript
   // Define a GET route
   app.get('/api/items', (req, res) => {
       res.send('Retrieve all items');
   });

   // Define a GET route for a specific item
   app.get('/api/items/:id', (req, res) => {
       const itemId = req.params.id;
       res.send(`Retrieve item with ID: ${itemId}`);
   });
   ```

2. **POST Method**: Create new data on the server.

   ```javascript
   // Define a POST route
   app.post('/api/items', (req, res) => {
       const newItem = req.body; // Get the new item from the request body
       res.status(201).send(`Item created: ${JSON.stringify(newItem)}`);
   });
   ```

3. **PUT Method**: Update existing data on the server.

   ```javascript
   // Define a PUT route
   app.put('/api/items/:id', (req, res) => {
       const itemId = req.params.id;
       const updatedItem = req.body;
       res.send(`Item with ID: ${itemId} updated with data: ${JSON.stringify(updatedItem)}`);
   });
   ```

4. **DELETE Method**: Delete data from the server.

   ```javascript
   // Define a DELETE route
   app.delete('/api/items/:id', (req, res) => {
       const itemId = req.params.id;
       res.send(`Item with ID: ${itemId} deleted`);
   });
   ```

#### Step 3: Combine Routes

Here’s how the full `app.js` file would look with all the routes combined:

```javascript
const express = require('express');
const app = express();

// Middleware to parse JSON request bodies
app.use(express.json());

const PORT = 3000;

// GET: Retrieve all items
app.get('/api/items', (req, res) => {
    res.send('Retrieve all items');
});

// GET: Retrieve a specific item
app.get('/api/items/:id', (req, res) => {
    const itemId = req.params.id;
    res.send(`Retrieve item with ID: ${itemId}`);
});

// POST: Create a new item
app.post('/api/items', (req, res) => {
    const newItem = req.body; // Get the new item from the request body
    res.status(201).send(`Item created: ${JSON.stringify(newItem)}`);
});

// PUT: Update an existing item
app.put('/api/items/:id', (req, res) => {
    const itemId = req.params.id;
    const updatedItem = req.body;
    res.send(`Item with ID: ${itemId} updated with data: ${JSON.stringify(updatedItem)}`);
});

// DELETE: Delete an item
app.delete('/api/items/:id', (req, res) => {
    const itemId = req.params.id;
    res.send(`Item with ID: ${itemId} deleted`);
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

#### Step 4: Test the Routes

1. **Start your server** by running:

   ```bash
   node app.js
   ```

2. **Test the routes** using a tool like Postman or curl:

   - **GET** all items: `GET http://localhost:3000/api/items`
   - **GET** a specific item: `GET http://localhost:3000/api/items/1`
   - **POST** a new item: `POST http://localhost:3000/api/items` with a JSON body, e.g., `{"name": "NewItem"}`.
   - **PUT** to update an item: `PUT http://localhost:3000/api/items/1` with a JSON body, e.g., `{"name": "UpdatedItem"}`.
   - **DELETE** an item: `DELETE http://localhost:3000/api/items/1`.

### Conclusion

By following these steps, you can define routes in Express and handle different HTTP methods effectively. This allows you to build a RESTful API or a web application that can interact with client requests, making it a powerful tool for backend development.