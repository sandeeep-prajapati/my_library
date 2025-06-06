### **Implement JavaScript Validation for Forms Rendered by Blade**

#### **Objective**  
Learn how to implement client-side JavaScript validation for forms in Blade templates, ensuring users provide correct and complete input before submitting.

---

### **1. Basic Form Validation**

#### **Example: Simple Contact Form Validation**

**Blade Template:**  
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Contact Us</h2>
        <form id="contactForm">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name">
            <span id="nameError" class="error"></span><br><br>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email">
            <span id="emailError" class="error"></span><br><br>

            <button type="submit">Submit</button>
        </form>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('contactForm').addEventListener('submit', function (e) {
        let isValid = true;

        // Validate Name
        const name = document.getElementById('name').value;
        const nameError = document.getElementById('nameError');
        if (name.trim() === "") {
            nameError.textContent = "Name is required.";
            isValid = false;
        } else {
            nameError.textContent = "";
        }

        // Validate Email
        const email = document.getElementById('email').value;
        const emailError = document.getElementById('emailError');
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailPattern.test(email)) {
            emailError.textContent = "Enter a valid email address.";
            isValid = false;
        } else {
            emailError.textContent = "";
        }

        // Prevent Form Submission if Validation Fails
        if (!isValid) {
            e.preventDefault();
        }
    });
</script>
@endpush
```

#### **Explanation:**  
- Use `addEventListener` to attach validation logic to the `submit` event.  
- Validate each input field and display error messages dynamically.  
- Prevent form submission with `e.preventDefault()` if validation fails.

---

### **2. Advanced Validation with Multiple Fields**

#### **Example: Registration Form**

**Blade Template:**  
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Register</h2>
        <form id="registrationForm">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username">
            <span id="usernameError" class="error"></span><br><br>

            <label for="password">Password:</label>
            <input type="password" id="password" name="password">
            <span id="passwordError" class="error"></span><br><br>

            <label for="confirmPassword">Confirm Password:</label>
            <input type="password" id="confirmPassword" name="confirmPassword">
            <span id="confirmPasswordError" class="error"></span><br><br>

            <button type="submit">Register</button>
        </form>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('registrationForm').addEventListener('submit', function (e) {
        let isValid = true;

        // Validate Username
        const username = document.getElementById('username').value;
        const usernameError = document.getElementById('usernameError');
        if (username.trim().length < 3) {
            usernameError.textContent = "Username must be at least 3 characters long.";
            isValid = false;
        } else {
            usernameError.textContent = "";
        }

        // Validate Password
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const passwordError = document.getElementById('passwordError');
        const confirmPasswordError = document.getElementById('confirmPasswordError');

        if (password.length < 6) {
            passwordError.textContent = "Password must be at least 6 characters.";
            isValid = false;
        } else {
            passwordError.textContent = "";
        }

        if (password !== confirmPassword) {
            confirmPasswordError.textContent = "Passwords do not match.";
            isValid = false;
        } else {
            confirmPasswordError.textContent = "";
        }

        if (!isValid) {
            e.preventDefault();
        }
    });
</script>
@endpush
```

#### **Explanation:**  
- Add validation for password length and matching passwords.  
- Check multiple conditions in a single field and provide appropriate error messages.

---

### **3. Real-Time Validation**

#### **Example: Email Validation on Input**

**Blade Template:**  
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Subscribe</h2>
        <form id="subscribeForm">
            <label for="email">Email:</label>
            <input type="email" id="email" name="email">
            <span id="emailFeedback" class="error"></span><br><br>

            <button type="submit">Subscribe</button>
        </form>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('email').addEventListener('input', function () {
        const email = this.value;
        const emailFeedback = document.getElementById('emailFeedback');
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

        if (emailPattern.test(email)) {
            emailFeedback.textContent = "Valid email!";
            emailFeedback.style.color = "green";
        } else {
            emailFeedback.textContent = "Invalid email format.";
            emailFeedback.style.color = "red";
        }
    });
</script>
@endpush
```

#### **Explanation:**  
- Use the `input` event to validate user input in real-time.  
- Provide instant feedback to users as they type.

---

### **4. Using HTML5 Validation with JavaScript**

#### **Example: Enhance Default Validation**

**Blade Template:**  
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Login</h2>
        <form id="loginForm" novalidate>
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            <span id="emailError" class="error"></span><br><br>

            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required>
            <span id="passwordError" class="error"></span><br><br>

            <button type="submit">Login</button>
        </form>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('loginForm').addEventListener('submit', function (e) {
        const email = document.getElementById('email');
        const password = document.getElementById('password');

        if (!email.checkValidity()) {
            document.getElementById('emailError').textContent = "Enter a valid email.";
            e.preventDefault();
        } else {
            document.getElementById('emailError').textContent = "";
        }

        if (!password.checkValidity()) {
            document.getElementById('passwordError').textContent = "Password is required.";
            e.preventDefault();
        } else {
            document.getElementById('passwordError').textContent = "";
        }
    });
</script>
@endpush
```

#### **Explanation:**  
- Leverage HTML5 validation methods like `checkValidity()` for better browser compatibility.  
- Use `novalidate` on the form to override default browser validation messages.

---

### **Best Practices for Form Validation**

1. **Keep Validation Messages Clear:**  
   Write concise and user-friendly error messages.

2. **Combine Client and Server Validation:**  
   Always validate data on the server to ensure security.

3. **Use CSS for Error Highlighting:**  
   Use styles to make errors visually distinct (e.g., red borders, icons).

4. **Optimize for Accessibility:**  
   Use `aria-live` regions for screen readers to announce validation feedback.

---
