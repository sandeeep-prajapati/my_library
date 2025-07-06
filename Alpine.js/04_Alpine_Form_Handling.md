# **Alpine.js Form Validation with Real-Time Email Check**

Here's a complete solution for binding form inputs with `x-model` and validating an email field in real-time:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        .error { color: red; font-size: 0.8rem; }
        input:invalid { border-color: red; }
        input:valid { border-color: green; }
    </style>
</head>
<body>
    <div 
        x-data="{ 
            form: {
                email: '',
                isValid: false
            },
            validateEmail(email) {
                const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                return re.test(email);
            }
        }"
        @submit.prevent="if(form.isValid) alert('Form submitted!')"
    >
        <form>
            <div>
                <label for="email">Email:</label>
                <input
                    type="email"
                    id="email"
                    x-model="form.email"
                    @input="form.isValid = validateEmail(form.email)"
                    :class="{ 'border-red-500': form.email && !form.isValid }"
                >
                <p 
                    x-show="form.email && !form.isValid"
                    class="error"
                >
                    Please enter a valid email address
                </p>
            </div>

            <button 
                type="submit"
                :disabled="!form.isValid"
                :class="{ 'opacity-50 cursor-not-allowed': !form.isValid }"
            >
                Submit
            </button>
        </form>

        <!-- Debug: Show current state -->
        <div x-text="`Email: ${form.email} | Valid: ${form.isValid}`"></div>
    </div>
</body>
</html>
```

## **Key Features Explained**

### **1. Form Data Binding (`x-model`)**
- `x-model="form.email"` creates a two-way binding between the input field and the `form.email` property.

### **2. Real-Time Validation**
- `@input="form.isValid = validateEmail(form.email)"` triggers validation on every keystroke.
- The `validateEmail()` function uses a regex pattern to check for a valid email format.

### **3. Error Handling**
- `x-show="form.email && !form.isValid"` displays the error message only when:
  - The field is not empty (`form.email` is truthy).
  - The email is invalid (`!form.isValid`).

### **4. Dynamic Styling**
- `:class="{ 'border-red-500': form.email && !form.isValid }"` adds a red border when invalid.
- The submit button is disabled (`:disabled="!form.isValid"`) until the email is valid.

### **5. Form Submission**
- `@submit.prevent` prevents the default form submission.
- The form only submits if `form.isValid` is `true`.

---

## **Challenge Extensions**

### **1. Add Password Validation**
```javascript
x-data="{
    form: {
        email: '',
        password: '',
        isValidEmail: false,
        isValidPassword: false
    },
    validateEmail(email) { /* ... */ },
    validatePassword(password) {
        return password.length >= 8;
    }
}"
```

### **2. Debounce Input (Avoid Excessive Validation)**
```html
<input 
    x-model.debounce.500ms="form.email"
    @input.debounce.500ms="form.isValid = validateEmail(form.email)"
>
```

### **3. Show Success State**
```html
<p 
    x-show="form.isValid"
    class="text-green-500"
>
    ✓ Valid email!
</p>
```

---

## **Why This Works**
- **Reactivity:** Alpine.js automatically updates the UI when `form.email` or `form.isValid` changes.
- **Declarative Validation:** No need for manual DOM manipulation—just define rules and let Alpine handle the rest.
- **User Experience:** Real-time feedback improves form usability.

Try pasting this into an HTML file and test with different email formats! 🚀