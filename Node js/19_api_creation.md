Creating RESTful APIs using Express.js involves setting up an Express application, defining routes that correspond to different HTTP methods (GET, POST, PUT, DELETE), and using middleware to handle requests and responses. Below is a step-by-step guide to building a simple RESTful API with Express.js.

### Step 1: Set Up Your Environment

1. **Create a New Directory for Your Project:**

   ```bash
   mkdir express-api
   cd express-api
   ```

2. **Initialize a New Node.js Project:**

   ```bash
   npm init -y
   ```

3. **Install Required Packages:**

   Install Express and any other necessary packages. In this example, we will also use Mongoose for MongoDB interactions.

   ```bash
   npm install express mongoose body-parser
   ```

### Step 2: Create Your Express Application

1. **Create an `app.js` File:**

   This file will be the main entry point for your application.

   ```javascript
   // app.js
   const express = require('express');
   const mongoose = require('mongoose');
   const bodyParser = require('body-parser');

   const app = express();
   const PORT = process.env.PORT || 3000;

   // Middleware
   app.use(bodyParser.json());

   // Connect to MongoDB
   mongoose.connect('mongodb://localhost:27017/express-api', { useNewUrlParser: true, useUnifiedTopology: true })
       .then(() => console.log('MongoDB connected'))
       .catch(err => console.log(err));

   // Start the server
   app.listen(PORT, () => {
       console.log(`Server is running on http://localhost:${PORT}`);
   });
   ```

### Step 3: Define a Model

For this example, let's create a simple **User** model.

1. **Create a `models` Directory and User Model:**

   ```bash
   mkdir models
   ```

   Create a `User.js` file inside the `models` directory.

   ```javascript
   // models/User.js
   const mongoose = require('mongoose');

   const userSchema = new mongoose.Schema({
       name: { type: String, required: true },
       email: { type: String, required: true, unique: true },
   });

   const User = mongoose.model('User', userSchema);

   module.exports = User;
   ```

### Step 4: Create RESTful Routes

Now, let's define the RESTful routes for our User resource.

1. **Create a `routes` Directory and User Routes:**

   ```bash
   mkdir routes
   ```

   Create a `userRoutes.js` file inside the `routes` directory.

   ```javascript
   // routes/userRoutes.js
   const express = require('express');
   const User = require('../models/User');

   const router = express.Router();

   // CREATE a new user
   router.post('/', async (req, res) => {
       try {
           const newUser = new User(req.body);
           const savedUser = await newUser.save();
           res.status(201).json(savedUser);
       } catch (err) {
           res.status(400).json({ message: err.message });
       }
   });

   // READ all users
   router.get('/', async (req, res) => {
       try {
           const users = await User.find();
           res.status(200).json(users);
       } catch (err) {
           res.status(500).json({ message: err.message });
       }
   });

   // READ a specific user
   router.get('/:id', async (req, res) => {
       try {
           const user = await User.findById(req.params.id);
           if (!user) return res.status(404).json({ message: 'User not found' });
           res.status(200).json(user);
       } catch (err) {
           res.status(500).json({ message: err.message });
       }
   });

   // UPDATE a user
   router.put('/:id', async (req, res) => {
       try {
           const updatedUser = await User.findByIdAndUpdate(req.params.id, req.body, { new: true });
           if (!updatedUser) return res.status(404).json({ message: 'User not found' });
           res.status(200).json(updatedUser);
       } catch (err) {
           res.status(400).json({ message: err.message });
       }
   });

   // DELETE a user
   router.delete('/:id', async (req, res) => {
       try {
           const deletedUser = await User.findByIdAndDelete(req.params.id);
           if (!deletedUser) return res.status(404).json({ message: 'User not found' });
           res.status(204).send();
       } catch (err) {
           res.status(500).json({ message: err.message });
       }
   });

   module.exports = router;
   ```

### Step 5: Use the Routes in Your Application

Now, you need to import the user routes and use them in your Express application.

1. **Update `app.js` to Include User Routes:**

   ```javascript
   // app.js
   const userRoutes = require('./routes/userRoutes');

   // Other existing code...

   app.use('/api/users', userRoutes);
   ```

### Step 6: Testing Your API

1. **Start Your Server:**

   ```bash
   node app.js
   ```

2. **Use Postman or any API testing tool** to test the following endpoints:

   - **Create a User**: `POST http://localhost:3000/api/users`
     - Request Body (JSON):
       ```json
       {
           "name": "John Doe",
           "email": "john@example.com"
       }
       ```
   
   - **Get All Users**: `GET http://localhost:3000/api/users`

   - **Get a Specific User**: `GET http://localhost:3000/api/users/:id` (replace `:id` with a valid user ID)

   - **Update a User**: `PUT http://localhost:3000/api/users/:id` (replace `:id` with a valid user ID)
     - Request Body (JSON):
       ```json
       {
           "name": "John Smith",
           "email": "johnsmith@example.com"
       }
       ```

   - **Delete a User**: `DELETE http://localhost:3000/api/users/:id` (replace `:id` with a valid user ID)

### Conclusion

You've successfully created a RESTful API using Express.js. This API includes routes for creating, reading, updating, and deleting users. You can expand on this by adding more features, such as input validation, authentication, or integrating other resources.