### **Use JavaScript to Handle Form Submissions Without Reloading the Page**

#### **Objective**  
Learn how to submit forms using JavaScript without reloading the page, utilizing `fetch` for AJAX requests. This approach improves user experience by allowing form submissions without a full page reload.

---

### **1. Basic Form Submission with `fetch`**

**Example: Contact Form Submission**

**Blade Template:**
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Contact Us</h2>
        <form id="contactForm">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required>
            <span id="nameError" class="error"></span><br><br>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            <span id="emailError" class="error"></span><br><br>

            <button type="submit">Submit</button>
        </form>

        <div id="responseMessage"></div>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('contactForm').addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the default form submission

        const form = this;
        const formData = new FormData(form);

        fetch('/submit-contact-form', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            // Handle successful submission
            if (data.success) {
                document.getElementById('responseMessage').textContent = 'Form submitted successfully!';
                document.getElementById('responseMessage').style.color = 'green';
                form.reset(); // Reset form after submission
            } else {
                document.getElementById('responseMessage').textContent = 'There was an error submitting the form.';
                document.getElementById('responseMessage').style.color = 'red';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('responseMessage').textContent = 'An error occurred. Please try again.';
            document.getElementById('responseMessage').style.color = 'red';
        });
    });
</script>
@endpush
```

#### **Explanation:**
- **Form Handling:**
  - We use JavaScript to capture the `submit` event of the form and prevent the default behavior with `e.preventDefault()`, ensuring the form doesn't refresh the page.
  - We gather form data using `new FormData(form)` and send it via a `POST` request using the `fetch` API.
- **AJAX Request:**
  - The `fetch` request sends the form data asynchronously to the server endpoint (`/submit-contact-form`).
  - The server responds with a JSON object indicating whether the submission was successful.
- **Response Handling:**
  - Based on the server response, we display a success or error message without reloading the page.

---

### **2. Server-Side Laravel Controller Example**

To handle the form submission on the server, create a route and controller method in Laravel.

**web.php (Routes):**
```php
use App\Http\Controllers\ContactFormController;

Route::post('/submit-contact-form', [ContactFormController::class, 'submit'])->name('submit.contact');
```

**ContactFormController.php:**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class ContactFormController extends Controller
{
    public function submit(Request $request)
    {
        // Basic validation
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'email' => 'required|email',
        ]);

        // Process the form (e.g., save to database, send email)
        // For now, we'll just simulate a success response.

        return response()->json(['success' => true]);
    }
}
```

#### **Explanation:**
- **Controller Method:**  
  - The `submit` method handles the `POST` request from the form submission.
  - It validates the data (you can add more validation logic if needed).
  - It returns a JSON response with `success: true` if everything is fine, which the JavaScript will handle.

---

### **3. Improving UX with Loading Spinner**

**Add a Loading Spinner during Form Submission:**

Modify the JavaScript to show a loading spinner while the form is being processed:

**Blade Template (Updated):**
```blade
@extends('layout')

@section('content')
    <div>
        <h2>Contact Us</h2>
        <form id="contactForm">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required>
            <span id="nameError" class="error"></span><br><br>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            <span id="emailError" class="error"></span><br><br>

            <button type="submit">Submit</button>
        </form>

        <div id="responseMessage"></div>
        <div id="loadingSpinner" style="display:none;">Submitting...</div>
    </div>
@endsection

@push('scripts')
<script>
    document.getElementById('contactForm').addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent default form submission

        const form = this;
        const formData = new FormData(form);
        const loadingSpinner = document.getElementById('loadingSpinner');
        const responseMessage = document.getElementById('responseMessage');

        // Show loading spinner
        loadingSpinner.style.display = 'block';
        responseMessage.textContent = ''; // Clear any previous messages

        fetch('/submit-contact-form', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner

            if (data.success) {
                responseMessage.textContent = 'Form submitted successfully!';
                responseMessage.style.color = 'green';
                form.reset();
            } else {
                responseMessage.textContent = 'There was an error submitting the form.';
                responseMessage.style.color = 'red';
            }
        })
        .catch(error => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner on error
            console.error('Error:', error);
            responseMessage.textContent = 'An error occurred. Please try again.';
            responseMessage.style.color = 'red';
        });
    });
</script>
@endpush
```

#### **Explanation of the Loading Spinner:**
- A `<div id="loadingSpinner">` is used to show a "Submitting..." message or a spinner during the form submission process.
- The spinner is displayed as soon as the form is submitted, and hidden when the form submission process is completed or if an error occurs.

---

### **4. Benefits of Using JavaScript for Form Submission Without Page Reload:**

- **Improved User Experience:**  
  No page reloads make the interaction smoother and faster.
  
- **Real-Time Feedback:**  
  You can show real-time feedback for validation errors, progress, or confirmation messages.
  
- **Asynchronous Processing:**  
  Server-side processing occurs without interrupting the user's interaction with the page.

---

This approach makes form submissions more dynamic and responsive by using JavaScript, providing a better user experience while maintaining server-side integrity.