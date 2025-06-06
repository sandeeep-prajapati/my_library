# **How to Upload and Manage Files in CodeIgniter?**  

File upload management is crucial for handling user-generated content like profile pictures, documents, and other media. CodeIgniter provides a robust **File Uploading Class** that simplifies the process.  

---

## **1. Configuring File Uploads**  

Ensure your `/writable/uploads` directory exists and is **writable**.  

```sh
mkdir -p writable/uploads
chmod 777 writable/uploads
```
✅ This allows CodeIgniter to store uploaded files properly.  

---

## **2. Creating the File Upload Form**  

📁 `app/Views/upload_form.php`  

```html
<!DOCTYPE html>
<html>
<head>
    <title>File Upload</title>
</head>
<body>
    <h2>Upload a File</h2>

    <?php if(session()->getFlashdata('error')): ?>
        <p style="color:red;"><?= session()->getFlashdata('error') ?></p>
    <?php endif; ?>

    <?php if(session()->getFlashdata('success')): ?>
        <p style="color:green;"><?= session()->getFlashdata('success') ?></p>
    <?php endif; ?>

    <form action="<?= base_url('/upload') ?>" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Upload</button>
    </form>
</body>
</html>
```
✅ Simple file upload form with error handling.  

---

## **3. Creating the Upload Controller**  

📁 `app/Controllers/UploadController.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class UploadController extends Controller
{
    public function index()
    {
        return view('upload_form');
    }

    public function upload()
    {
        $file = $this->request->getFile('file');

        if ($file->isValid() && !$file->hasMoved()) {
            $newName = $file->getRandomName(); // Generate a unique filename
            $file->move(WRITEPATH . 'uploads', $newName);

            session()->setFlashdata('success', 'File uploaded successfully!');
            return redirect()->to('/upload');
        } else {
            session()->setFlashdata('error', 'Failed to upload file.');
            return redirect()->to('/upload');
        }
    }
}
```
✅ **Handles file uploads**, renames files to avoid conflicts, and moves them to the upload directory.  

---

## **4. Defining Routes for File Uploading**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/upload', 'UploadController::index');
$routes->post('/upload', 'UploadController::upload');
```
✅ Adds routes for the upload form and processing.  

---

## **5. Displaying Uploaded Files**  

Modify the `UploadController.php` to list uploaded files:  

```php
public function listFiles()
{
    $files = array_diff(scandir(WRITEPATH . 'uploads'), ['.', '..']);
    return view('file_list', ['files' => $files]);
}
```
✅ Fetches uploaded files from the `uploads` directory.  

---

### **Creating the View for Displaying Files**  

📁 `app/Views/file_list.php`  

```html
<!DOCTYPE html>
<html>
<head>
    <title>Uploaded Files</title>
</head>
<body>
    <h2>Uploaded Files</h2>

    <ul>
        <?php foreach($files as $file): ?>
            <li><a href="<?= base_url('download/' . $file) ?>"><?= $file ?></a></li>
        <?php endforeach; ?>
    </ul>
</body>
</html>
```
✅ Lists all uploaded files with download links.  

---

## **6. Downloading Files**  

Modify `UploadController.php` to add a download function:  

```php
public function download($filename)
{
    $filePath = WRITEPATH . 'uploads/' . $filename;

    if (file_exists($filePath)) {
        return $this->response->download($filePath, null);
    } else {
        return redirect()->to('/files')->with('error', 'File not found');
    }
}
```
✅ Enables file downloads via a route.  

---

### **Add Route for File Download**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/files', 'UploadController::listFiles');
$routes->get('/download/(:any)', 'UploadController::download/$1');
```
✅ Users can now view and download uploaded files.  

---

## **7. Deleting Uploaded Files**  

Modify `UploadController.php` to allow file deletion:  

```php
public function delete($filename)
{
    $filePath = WRITEPATH . 'uploads/' . $filename;

    if (file_exists($filePath)) {
        unlink($filePath);
        return redirect()->to('/files')->with('success', 'File deleted successfully');
    } else {
        return redirect()->to('/files')->with('error', 'File not found');
    }
}
```
✅ Deletes files from the `uploads` directory.  

---

### **Modify `file_list.php` to Add Delete Option**  

```html
<li>
    <a href="<?= base_url('download/' . $file) ?>"><?= $file ?></a> 
    | <a href="<?= base_url('delete/' . $file) ?>" style="color:red;">Delete</a>
</li>
```
✅ Users can now delete uploaded files.  

---

### **Add Route for Deleting Files**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/delete/(:any)', 'UploadController::delete/$1');
```
✅ Adds a route for deleting uploaded files.  

---

## **Final Thoughts**  

✔ **File Uploading:** Users can upload files.  
✔ **File Listing:** Displays uploaded files.  
✔ **File Downloading:** Users can download files.  
✔ **File Deletion:** Users can delete files.  

🚀 **Next Step:** Do you want to add **image resizing** or **file type validation**? 😊