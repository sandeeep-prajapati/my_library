Serving static files like HTML, CSS, and JavaScript in an Express application is straightforward using the built-in middleware provided by Express. Here's how to set it up step-by-step:

### Step 1: Set Up Your Project

If you haven't already set up an Express project, you can do so by following these steps:

1. **Create a new directory** for your project and navigate into it:

   ```bash
   mkdir my-express-app
   cd my-express-app
   ```

2. **Initialize a new Node.js project**:

   ```bash
   npm init -y
   ```

3. **Install Express**:

   ```bash
   npm install express
   ```

### Step 2: Organize Your Project Structure

Create a folder structure for your static files. For example:

```
my-express-app/
│
├── public/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── scripts.js
│   └── index.html
│
└── app.js
```

### Step 3: Create Your Static Files

Create a simple `index.html` file in the `public` directory:

```html
<!-- public/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/styles.css">
    <title>Static Files Example</title>
</head>
<body>
    <h1>Hello, Express!</h1>
    <script src="js/scripts.js"></script>
</body>
</html>
```

Create a simple CSS file:

```css
/* public/css/styles.css */
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
    text-align: center;
    padding: 50px;
}
```

And a JavaScript file:

```javascript
// public/js/scripts.js
document.addEventListener('DOMContentLoaded', () => {
    console.log('JavaScript is loaded!');
});
```

### Step 4: Serve Static Files in Express

In your `app.js` file, you can set up Express to serve the static files:

```javascript
const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 5: Run Your Application

1. Start your Express server:

   ```bash
   node app.js
   ```

2. Open your browser and navigate to `http://localhost:3000`. You should see your HTML page styled with the CSS and with the JavaScript running.

### Summary

By using `express.static`, you can easily serve static files from a specified directory. This is a common practice in web applications to serve assets like images, stylesheets, and scripts. You can adjust the directory served by modifying the path in `app.use()`, and you can even serve multiple directories if needed.