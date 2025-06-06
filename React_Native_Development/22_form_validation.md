
---

## **Steps to Implement Form Validation**

### 1. **Setup State for Input Values and Errors**
- Use `useState` to manage the form's input values and error messages.

---

### 2. **Validation Logic**
- Define validation rules for your inputs (e.g., required fields, email format, etc.).

---

### 3. **Show Error Messages**
- Display error messages conditionally based on validation results.

---

### 4. **Implementation Example**

Here’s a complete example:

```jsx
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';

const FormValidationExample = () => {
  const [form, setForm] = useState({ name: '', email: '' }); // Input values
  const [errors, setErrors] = useState({}); // Error messages

  // Validation function
  const validate = () => {
    let valid = true;
    let newErrors = {};

    // Name validation
    if (!form.name.trim()) {
      newErrors.name = 'Name is required';
      valid = false;
    }

    // Email validation
    if (!form.email.trim()) {
      newErrors.email = 'Email is required';
      valid = false;
    } else if (!/\S+@\S+\.\S+/.test(form.email)) {
      newErrors.email = 'Email is not valid';
      valid = false;
    }

    setErrors(newErrors); // Update errors state
    return valid;
  };

  // Submit handler
  const handleSubmit = () => {
    if (validate()) {
      alert('Form submitted successfully!');
      // Handle form submission (e.g., API call)
    }
  };

  // Input change handler
  const handleChange = (field, value) => {
    setForm({ ...form, [field]: value });
    setErrors({ ...errors, [field]: '' }); // Clear error for this field
  };

  return (
    <View style={styles.container}>
      {/* Name Input */}
      <TextInput
        style={styles.input}
        placeholder="Name"
        value={form.name}
        onChangeText={(value) => handleChange('name', value)}
      />
      {errors.name && <Text style={styles.error}>{errors.name}</Text>}

      {/* Email Input */}
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={form.email}
        onChangeText={(value) => handleChange('email', value)}
        keyboardType="email-address"
      />
      {errors.email && <Text style={styles.error}>{errors.email}</Text>}

      {/* Submit Button */}
      <Button title="Submit" onPress={handleSubmit} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
    marginBottom: 10,
  },
  error: {
    color: 'red',
    marginBottom: 10,
  },
});

export default FormValidationExample;
```

---

### **Explanation of the Code**

1. **State Management**
   - `form`: Holds the input values (e.g., name, email).
   - `errors`: Holds error messages for each input field.

2. **Validation Rules**
   - Check if the input is empty (required fields).
   - Use regex for email validation.

3. **Error Display**
   - Errors are stored in the `errors` object, and the corresponding message is displayed below the input field.

4. **Input Handling**
   - The `handleChange` function updates the form state and clears any existing error for the field.

5. **Submit Handling**
   - The `handleSubmit` function validates the inputs and alerts the user if the form is valid.

---

### **Extending Validation**
You can extend the validation logic to include:
- Minimum and maximum length checks:
  ```javascript
  if (form.name.length < 3) {
    newErrors.name = 'Name must be at least 3 characters';
  }
  ```
- Password complexity:
  ```javascript
  if (!/^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{6,}$/.test(form.password)) {
    newErrors.password = 'Password must contain at least one uppercase letter, one number, and be at least 6 characters long';
  }
  ```
- Matching fields (e.g., confirm password).

---
