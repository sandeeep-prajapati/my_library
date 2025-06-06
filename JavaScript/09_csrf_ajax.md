### **Configure JavaScript to Include Laravel's CSRF Token in AJAX Requests**

Laravel includes a CSRF (Cross-Site Request Forgery) token by default in your Blade views to secure your forms and AJAX requests. When using JavaScript to make AJAX requests, you need to ensure that the CSRF token is included in the request headers to prevent CSRF attacks.

Here’s how you can configure your JavaScript to automatically include the CSRF token in AJAX requests using the `fetch` API or `Axios`.

---

### **1. Set Up CSRF Token in Your Blade Template**

In your Blade template, you can access the CSRF token from the Laravel Blade syntax. Laravel provides a convenient way to inject the CSRF token into your HTML headers by using the `@csrf` directive.

#### **Blade Template (e.g., `resources/views/layout.blade.php`)**
```blade
<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
</head>
```

This will place the CSRF token in a `<meta>` tag in the HTML `<head>` section. The JavaScript can later access this token for making secure requests.

---

### **2. Include CSRF Token in JavaScript Using `fetch`**

You can then use the token in your JavaScript code. For this example, let's use the `fetch` API to make an AJAX request that includes the CSRF token.

#### **JavaScript Example Using `fetch`**
```javascript
// Get the CSRF token from the meta tag
const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

// Example function to submit data via AJAX
function submitFormData(formData) {
    fetch('/submit-form', {
        method: 'POST', // HTTP method
        headers: {
            'Content-Type': 'application/json', // Sending JSON data
            'X-CSRF-TOKEN': csrfToken // Include the CSRF token in the header
        },
        body: JSON.stringify(formData) // Convert form data to JSON string
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
```

#### **Explanation:**
- The CSRF token is fetched from the `<meta>` tag (`document.querySelector('meta[name="csrf-token"]')`).
- The `X-CSRF-TOKEN` header is added to the AJAX request with the CSRF token as its value.
- The `Content-Type: 'application/json'` is specified to indicate that the request body contains JSON data.

---

### **3. Include CSRF Token in JavaScript Using Axios**

If you are using the `Axios` library for making AJAX requests, Axios automatically supports setting the CSRF token in the request headers.

#### **JavaScript Example Using Axios**
```javascript
// Set up Axios default headers to include the CSRF token
axios.defaults.headers.common['X-CSRF-TOKEN'] = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

// Example function to submit data via Axios
function submitFormData(formData) {
    axios.post('/submit-form', formData)
        .then(response => {
            console.log('Success:', response.data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
```

#### **Explanation:**
- `axios.defaults.headers.common['X-CSRF-TOKEN']` is used to globally set the CSRF token for all subsequent requests.
- Once the token is set globally, you don’t need to manually add the token for each individual request.
- The `axios.post()` method is used to submit the form data to the server.

---

### **4. Handling CSRF Token Expiration**

If the CSRF token expires or is invalid, Laravel will respond with a `419` HTTP status code (Page Expired). You can catch this error in JavaScript and handle it appropriately, like reloading the page or prompting the user to re-authenticate.

#### **Handling Expired CSRF Token Example**
```javascript
fetch('/submit-form', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-TOKEN': csrfToken
    },
    body: JSON.stringify(formData)
})
.then(response => {
    if (!response.ok) {
        if (response.status === 419) {
            // Handle CSRF token expiration
            alert('Session expired. Please refresh the page.');
            location.reload(); // Reload the page to get a new CSRF token
        }
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(data => {
    console.log('Form submitted successfully', data);
})
.catch(error => {
    console.error('There was a problem with the fetch operation:', error);
});
```

---

### **5. Laravel CSRF Token Handling in Form Requests (Optional)**

If you’re using Blade to handle the form rendering and submitting, Laravel automatically includes the CSRF token in every form rendered by Blade. You can manually include the CSRF token in forms using:

```blade
<form method="POST" action="/submit-form">
    @csrf  <!-- Automatically includes the CSRF token -->
    <!-- Form fields go here -->
    <button type="submit">Submit</button>
</form>
```

In AJAX requests, however, you need to manually add the CSRF token to your headers, as shown in the above examples.

---

### **Conclusion**

By including the CSRF token in your AJAX requests, you ensure that your application is protected against CSRF attacks while still allowing for dynamic, AJAX-driven user interactions. Whether you are using the `fetch` API or a library like `Axios`, this method is crucial to keep your Laravel application secure.