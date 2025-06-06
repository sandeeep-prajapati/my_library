In Express, handling HTTP requests and sending responses is a fundamental part of building web applications. Here’s a step-by-step guide on how to manage different types of HTTP requests (GET, POST, PUT, DELETE) and send appropriate responses.

### Setting Up an Express Application

1. **Install Express**: If you haven’t installed Express yet, you can do so using npm.

   ```bash
   npm install express
   ```

2. **Create a Basic Express Server**:

   ```javascript
   const express = require('express');
   const app = express();
   const PORT = 3000;

   // Middleware to parse JSON request bodies
   app.use(express.json());

   app.listen(PORT, () => {
       console.log(`Server is running on http://localhost:${PORT}`);
   });
   ```

### Handling HTTP Requests

#### 1. **GET Requests**

GET requests are used to retrieve data from the server.

```javascript
app.get('/api/items', (req, res) => {
    // Sample data to send in response
    const items = [
        { id: 1, name: 'Item One' },
        { id: 2, name: 'Item Two' },
    ];
    res.json(items); // Send a JSON response
});
```

#### 2. **POST Requests**

POST requests are used to create new resources on the server.

```javascript
app.post('/api/items', (req, res) => {
    const newItem = req.body; // Get the new item from the request body
    // Here, you would typically save the new item to a database
    res.status(201).json({ message: 'Item created successfully', item: newItem }); // Send a success response
});
```

#### 3. **PUT Requests**

PUT requests are used to update existing resources.

```javascript
app.put('/api/items/:id', (req, res) => {
    const itemId = req.params.id; // Get the item ID from the URL
    const updatedItem = req.body; // Get the updated item from the request body
    // Here, you would typically update the item in the database
    res.json({ message: `Item ${itemId} updated successfully`, item: updatedItem }); // Send a success response
});
```

#### 4. **DELETE Requests**

DELETE requests are used to delete resources from the server.

```javascript
app.delete('/api/items/:id', (req, res) => {
    const itemId = req.params.id; // Get the item ID from the URL
    // Here, you would typically delete the item from the database
    res.json({ message: `Item ${itemId} deleted successfully` }); // Send a success response
});
```

### Complete Example

Here’s a complete example that combines all the above methods in a single Express application:

```javascript
const express = require('express');
const app = express();
const PORT = 3000;

app.use(express.json());

// Sample in-memory data storage
let items = [
    { id: 1, name: 'Item One' },
    { id: 2, name: 'Item Two' },
];

// GET request
app.get('/api/items', (req, res) => {
    res.json(items);
});

// POST request
app.post('/api/items', (req, res) => {
    const newItem = req.body;
    newItem.id = items.length + 1; // Assign a new ID
    items.push(newItem); // Add to the in-memory data
    res.status(201).json({ message: 'Item created successfully', item: newItem });
});

// PUT request
app.put('/api/items/:id', (req, res) => {
    const itemId = parseInt(req.params.id);
    const updatedItem = req.body;
    items = items.map(item => (item.id === itemId ? { ...item, ...updatedItem } : item));
    res.json({ message: `Item ${itemId} updated successfully`, item: updatedItem });
});

// DELETE request
app.delete('/api/items/:id', (req, res) => {
    const itemId = parseInt(req.params.id);
    items = items.filter(item => item.id !== itemId);
    res.json({ message: `Item ${itemId} deleted successfully` });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Testing the API

You can use tools like [Postman](https://www.postman.com/) or [curl](https://curl.se/) to test your API endpoints.

1. **GET all items**: `GET http://localhost:3000/api/items`
2. **Create a new item**: `POST http://localhost:3000/api/items` with body `{"name": "Item Three"}`
3. **Update an item**: `PUT http://localhost:3000/api/items/1` with body `{"name": "Updated Item One"}`
4. **Delete an item**: `DELETE http://localhost:3000/api/items/1`

### Conclusion

With this setup, you can handle various HTTP requests and send responses using Express. This forms the backbone of any RESTful API, allowing you to manage resources effectively.