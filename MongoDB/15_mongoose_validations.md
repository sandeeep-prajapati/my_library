Setting up field validations in Mongoose is crucial for ensuring data integrity and maintaining the quality of your MongoDB documents. Mongoose provides several built-in validation features, and you can also create custom validations to meet specific requirements.

### 1. **Built-in Validations**

Mongoose supports various built-in validations that you can apply to your schema fields. Here are some common validations:

- **Required Validation**: Ensures that a field is not empty.
- **Min and Max Length**: Checks the length of strings and arrays.
- **Min and Max Value**: Checks the numerical values of fields.
- **Enums**: Validates that a field's value is one of the specified values.
- **Unique**: Ensures that the value of a field is unique across the collection (note that this is not enforced by Mongoose itself but by the database through indexes).

#### Example Schema with Built-in Validations

```javascript
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true, // Field is required
        minlength: 3, // Minimum length of 3
        maxlength: 30, // Maximum length of 30
        unique: true // Value must be unique
    },
    email: {
        type: String,
        required: true,
        match: /.+\@.+\..+/ // Basic email format validation
    },
    age: {
        type: Number,
        min: 0, // Minimum value of 0
        max: 120 // Maximum value of 120
    },
    role: {
        type: String,
        enum: ['user', 'admin'], // Role must be either 'user' or 'admin'
        default: 'user' // Default value if none is provided
    }
});

const User = mongoose.model('User', userSchema);
```

### 2. **Custom Validations**

Mongoose allows you to define custom validation functions for more complex validation scenarios that cannot be handled by built-in validators. You can implement custom validators by providing a function that returns `true` if validation passes or a string/message if it fails.

#### Example of Custom Validation

```javascript
const userSchema = new mongoose.Schema({
    password: {
        type: String,
        required: true,
        validate: {
            validator: function(v) {
                // Custom validation logic: password must contain at least one number and one special character
                return /[0-9]/.test(v) && /[!@#$%^&*]/.test(v);
            },
            message: props => `${props.value} is not a valid password! Password must contain at least one number and one special character.`
        }
    }
});
```

### 3. **Asynchronous Validations**

For validations that require asynchronous checks (e.g., checking if a username already exists in the database), you can define an asynchronous validator using a `Promise`.

#### Example of Asynchronous Custom Validation

```javascript
userSchema.path('username').validate(async function(value) {
    const count = await User.countDocuments({ username: value });
    return count === 0; // Username must be unique
}, 'Username already exists!');
```

### 4. **Error Handling**

When validation fails, Mongoose will throw a validation error. You can handle these errors in your application code as follows:

```javascript
const newUser = new User({ username: 'test', email: 'test@example.com', password: 'password123' });

newUser.save()
    .then(() => {
        console.log('User saved successfully!');
    })
    .catch(err => {
        if (err.name === 'ValidationError') {
            console.log('Validation Error:', err.message);
            for (const key in err.errors) {
                console.log(err.errors[key].message);
            }
        } else {
            console.error('Error:', err);
        }
    });
```

### Conclusion

By implementing both built-in and custom validations in your Mongoose schemas, you can ensure data integrity and enforce rules that maintain the quality of your MongoDB documents. Proper error handling for validation errors will help provide informative feedback to users and developers, making your application more robust.