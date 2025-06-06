In Express, you can easily access query parameters and URL parameters (also known as route parameters) to retrieve data sent by clients through the URL. Here's how to do both:

### Accessing Query Parameters

Query parameters are key-value pairs that appear in the URL after the `?` character. They are typically used to filter or paginate data.

#### Example

Given the following URL: 

```
http://localhost:3000/api/items?category=books&sort=asc
```

1. **Define a Route to Handle the Request**:

```javascript
const express = require('express');
const app = express();

app.get('/api/items', (req, res) => {
    // Accessing query parameters using req.query
    const category = req.query.category; // "books"
    const sort = req.query.sort; // "asc"
    
    res.send(`Category: ${category}, Sort: ${sort}`);
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

2. **Test the Route**:

Start your server and navigate to `http://localhost:3000/api/items?category=books&sort=asc`. You should see the response:

```
Category: books, Sort: asc
```

### Accessing URL Parameters (Route Parameters)

URL parameters are defined in the route path and are typically used to capture dynamic segments of the URL.

#### Example

Given the following URL: 

```
http://localhost:3000/api/items/42
```

1. **Define a Route with a URL Parameter**:

```javascript
const express = require('express');
const app = express();

app.get('/api/items/:id', (req, res) => {
    // Accessing URL parameters using req.params
    const itemId = req.params.id; // "42"
    
    res.send(`Item ID: ${itemId}`);
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

2. **Test the Route**:

Start your server and navigate to `http://localhost:3000/api/items/42`. You should see the response:

```
Item ID: 42
```

### Combining Query and URL Parameters

You can also combine both query parameters and URL parameters in the same route.

#### Example

```javascript
const express = require('express');
const app = express();

app.get('/api/items/:id', (req, res) => {
    const itemId = req.params.id; // URL parameter
    const category = req.query.category; // Query parameter
    
    res.send(`Item ID: ${itemId}, Category: ${category}`);
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

#### Test the Combined Route

Using the URL:

```
http://localhost:3000/api/items/42?category=books
```

The response would be:

```
Item ID: 42, Category: books
```

### Conclusion

Using `req.query` for query parameters and `req.params` for URL parameters in Express makes it straightforward to handle client data in your routes. This allows you to build dynamic and interactive web applications efficiently.