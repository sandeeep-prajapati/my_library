To show a progress bar for file uploads in a Laravel project using JavaScript, you can use the `XMLHttpRequest` (XHR) object for tracking the upload progress. The progress bar will dynamically update as the file is uploaded, providing visual feedback to the user.

Here’s a complete guide to implement a progress bar for file uploads in a Laravel and JavaScript setup.

### **Step 1: Update the Blade Template for Progress Bar**

First, we'll modify the Blade template to include a progress bar that will be updated dynamically as the file uploads.

```html
<!-- resources/views/file-upload.blade.php -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload with Progress Bar</title>
    <link rel="stylesheet" href="{{ asset('css/app.css') }}">
</head>
<body>
    <div class="container mt-5">
        <h2>Upload File with Progress Bar</h2>

        <!-- File Upload Form -->
        <input type="file" id="fileInput" class="form-control mb-3" accept="image/*">
        
        <!-- Progress Bar -->
        <div id="progressContainer" class="mt-3" style="display:none;">
            <label for="progressBar">Uploading...</label>
            <progress id="progressBar" value="0" max="100" style="width: 100%;"></progress>
        </div>

        <!-- Submit button -->
        <button id="uploadBtn" class="btn btn-primary mt-3">Upload</button>
    </div>

    <script src="{{ asset('js/app.js') }}"></script>
</body>
</html>
```

### **Explanation of Blade Template Changes:**

1. **File Input**: The file input (`<input type="file">`) allows the user to select a file for uploading.
2. **Progress Bar**: The `<progress>` element is used to display the upload progress. Initially, it is hidden (`display:none;`), and it will only be shown once the upload starts.
3. **Upload Button**: The button triggers the file upload when clicked.

---

### **Step 2: JavaScript for Handling File Upload with Progress Bar**

Now, we will use JavaScript to handle the file upload and update the progress bar.

```javascript
// resources/js/app.js

document.getElementById('uploadBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file first');
        return;
    }

    // Show the progress container and reset the progress bar
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    progressContainer.style.display = 'block';
    progressBar.value = 0; // Reset progress bar

    // Create FormData object to hold the file for AJAX upload
    const formData = new FormData();
    formData.append('file', file);

    // Create XMLHttpRequest to send the file
    const xhr = new XMLHttpRequest();
    
    // Event listener for tracking upload progress
    xhr.upload.addEventListener('progress', function(event) {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            progressBar.value = percentComplete;
        }
    });

    // Event listener for when the upload is complete
    xhr.onload = function() {
        if (xhr.status == 200) {
            // File uploaded successfully
            alert('File uploaded successfully!');
            progressBar.value = 100;
        } else {
            // File upload failed
            alert('Error uploading file.');
        }
        progressContainer.style.display = 'none'; // Hide progress bar after upload
    };

    // Event listener for handling errors during upload
    xhr.onerror = function() {
        alert('An error occurred during file upload.');
        progressContainer.style.display = 'none'; // Hide progress bar on error
    };

    // Send the request
    xhr.open('POST', '{{ route('file.upload') }}', true);
    xhr.setRequestHeader('X-CSRF-TOKEN', '{{ csrf_token() }}'); // CSRF Token
    xhr.send(formData);
});
```

### **Explanation of JavaScript Logic:**

1. **File Selection**: When the upload button is clicked, it checks if a file is selected. If not, it alerts the user.
2. **Progress Bar**: 
   - The progress container is shown, and the progress bar is reset to 0 before starting the upload.
   - The `XMLHttpRequest` is used to send the file asynchronously.
   - The `progress` event of `xhr.upload` is used to track the upload progress, updating the progress bar as the file is uploaded.
3. **Upload Success and Error Handling**:
   - When the upload is complete, the status is checked. If successful, the progress bar reaches 100%, and an alert is shown.
   - If an error occurs during the upload, an error message is displayed.

---

### **Step 3: Update Controller for Handling File Upload**

In the backend, ensure the controller can handle the file upload properly. Here’s the updated controller.

```php
// app/Http/Controllers/FileUploadController.php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class FileUploadController extends Controller
{
    public function upload(Request $request)
    {
        // Validate the file
        $request->validate([
            'file' => 'required|file|mimes:jpg,jpeg,png,pdf|max:2048', // Add more types as needed
        ]);

        // Store the file
        $path = $request->file('file')->store('uploads', 'public');

        // Return response to AJAX request
        return response()->json(['success' => true, 'path' => $path]);
    }
}
```

### **Step 4: Define the Route for File Upload**

Add the route to handle the file upload request in `web.php`:

```php
// routes/web.php

use App\Http\Controllers\FileUploadController;

Route::post('/upload-file', [FileUploadController::class, 'upload'])->name('file.upload');
```

---

### **Step 5: Ensure File Storage Configuration**

Make sure you’ve set up symbolic linking to access uploaded files publicly:

```bash
php artisan storage:link
```

This command creates a symbolic link from `public/storage` to `storage/app/public` so you can access uploaded files via URLs.

---

### **Step 6: Test the File Upload with Progress Bar**

1. **Start the Laravel server** by running `php artisan serve`.
2. **Navigate to the file upload page** and try uploading a file.
3. The progress bar should update as the file is uploaded, and once the upload is complete, it should show a success message.

---

### **Conclusion**

This solution demonstrates how to implement a file upload system with a progress bar in Laravel using JavaScript. The progress bar dynamically updates as the file uploads, providing feedback to the user. The backend handles the file storage, and the AJAX request ensures the upload happens without refreshing the page.