Handling form submissions and parsing form data in Express is straightforward, thanks to middleware that processes incoming request bodies. Below is a step-by-step guide on how to set this up, including parsing form data using both URL-encoded and multipart/form-data formats.

### Step 1: Set Up Your Express Application

1. **Create a new Express application** (if you haven't already):

   ```bash
   mkdir form-submission-app
   cd form-submission-app
   npm init -y
   npm install express body-parser multer
   ```

### Step 2: Set Up the Server

Create a file named `app.js` and set up a basic Express server:

```javascript
const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');

const app = express();
const PORT = 3000;

// Set up body-parser middleware for URL-encoded data
app.use(bodyParser.urlencoded({ extended: true }));

// Set up multer middleware for handling multipart/form-data
const upload = multer({ dest: 'uploads/' });

// Create a route to serve the HTML form
app.get('/', (req, res) => {
    res.send(`
        <h1>Form Submission</h1>
        <form action="/submit" method="POST">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required><br><br>
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required><br><br>
            <button type="submit">Submit</button>
        </form>
    `);
});

// Create a route to handle form submissions
app.post('/submit', (req, res) => {
    const { name, email } = req.body; // Access the parsed form data
    res.send(`Form submitted successfully! Name: ${name}, Email: ${email}`);
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

### Step 3: Running the Application

1. Start your Express server:

   ```bash
   node app.js
   ```

2. Open your browser and navigate to `http://localhost:3000`. You will see a form with fields for name and email.

3. Fill in the form and submit it. You should see a message confirming the submission.

### Step 4: Handling File Uploads (Optional)

If you want to handle file uploads in addition to regular form data, you can use the `multer` middleware. Here’s how to modify the form to include a file upload:

1. **Update the HTML form** to allow file uploads:

   ```javascript
   app.get('/', (req, res) => {
       res.send(`
           <h1>Form Submission</h1>
           <form action="/submit" method="POST" enctype="multipart/form-data">
               <label for="name">Name:</label>
               <input type="text" id="name" name="name" required><br><br>
               <label for="email">Email:</label>
               <input type="email" id="email" name="email" required><br><br>
               <label for="file">File:</label>
               <input type="file" id="file" name="file"><br><br>
               <button type="submit">Submit</button>
           </form>
       `);
   });
   ```

2. **Update the form submission route** to handle file uploads:

   ```javascript
   app.post('/submit', upload.single('file'), (req, res) => {
       const { name, email } = req.body; // Access the parsed form data
       const file = req.file; // Access the uploaded file
       res.send(`Form submitted successfully! Name: ${name}, Email: ${email}, File: ${file ? file.originalname : 'No file uploaded'}`);
   });
   ```

### Summary

1. **Form Submission**: You set up a basic HTML form and created a route to handle the form submission.
2. **Parsing Form Data**: You used `body-parser` to handle URL-encoded data and `multer` for handling multipart/form-data (file uploads).
3. **Accessing Data**: You accessed submitted data from `req.body` for text fields and `req.file` for uploaded files.

This setup allows you to handle form submissions and parse form data effectively in your Express applications.