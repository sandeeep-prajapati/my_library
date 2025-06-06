Template engines are tools that enable developers to generate HTML dynamically by combining templates with data. They allow you to create reusable HTML components, implement conditional rendering, and loop through data collections, making it easier to maintain and update the front-end of web applications.

### Popular Template Engines for Express

1. **EJS (Embedded JavaScript)**: A simple templating language that lets you generate HTML with plain JavaScript.
2. **Pug**: A high-performance template engine that uses a clean syntax and indentation instead of traditional HTML tags.

### Using EJS with Express

#### Step 1: Set Up Your Project

If you haven’t already, set up your Express project:

```bash
mkdir my-express-app
cd my-express-app
npm init -y
npm install express ejs
```

#### Step 2: Configure EJS in Express

1. **Create a basic Express application**:

   Create a file named `app.js`:

   ```javascript
   const express = require('express');
   const app = express();
   const PORT = 3000;

   // Set EJS as the templating engine
   app.set('view engine', 'ejs');

   // Define a route
   app.get('/', (req, res) => {
       res.render('index', { title: 'Home Page', message: 'Welcome to EJS!' });
   });

   app.listen(PORT, () => {
       console.log(`Server is running on http://localhost:${PORT}`);
   });
   ```

2. **Create a views directory**:

   Create a directory named `views` in the root of your project:

   ```
   my-express-app/
   ├── app.js
   └── views/
       └── index.ejs
   ```

3. **Create an EJS template**:

   In the `views` directory, create an `index.ejs` file:

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title><%= title %></title>
   </head>
   <body>
       <h1><%= message %></h1>
   </body>
   </html>
   ```

#### Step 3: Run Your Application

1. Start your server:

   ```bash
   node app.js
   ```

2. Open your browser and navigate to `http://localhost:3000`. You should see the message “Welcome to EJS!” displayed on the page.

### Using Pug with Express

#### Step 1: Install Pug

If you want to use Pug instead of EJS, you’ll need to install it:

```bash
npm install pug
```

#### Step 2: Configure Pug in Express

1. Modify your `app.js` file to use Pug:

   ```javascript
   const express = require('express');
   const app = express();
   const PORT = 3000;

   // Set Pug as the templating engine
   app.set('view engine', 'pug');

   // Define a route
   app.get('/', (req, res) => {
       res.render('index', { title: 'Home Page', message: 'Welcome to Pug!' });
   });

   app.listen(PORT, () => {
       console.log(`Server is running on http://localhost:${PORT}`);
   });
   ```

2. **Create a Pug template**:

   Create a directory named `views` (if you haven’t already) and add an `index.pug` file:

   ```
   my-express-app/
   ├── app.js
   └── views/
       └── index.pug
   ```

   Add the following content to `index.pug`:

   ```pug
   doctype html
   html(lang="en")
       head
           meta(charset="UTF-8")
           meta(name="viewport", content="width=device-width, initial-scale=1.0")
           title #{title}
       body
           h1 #{message}
   ```

#### Step 3: Run Your Application

1. Start your server again:

   ```bash
   node app.js
   ```

2. Open your browser and navigate to `http://localhost:3000`. You should see the message “Welcome to Pug!” displayed on the page.

### Summary

- **Template engines** allow you to generate HTML dynamically using templates and data, making web development more efficient.
- **EJS** and **Pug** are popular choices for Express applications.
- To use a template engine, you need to set it up in your Express application, create templates, and render them with data.

By integrating a template engine with Express, you can easily create dynamic web pages that respond to user input and data changes.