# **How to Resize, Crop, and Watermark Images Using CodeIgniter’s Image Library?**  

CodeIgniter provides a powerful **Image Manipulation Library** to resize, crop, and add watermarks to images. In this guide, we will cover:  
✅ **Resizing an image**  
✅ **Cropping an image**  
✅ **Adding a watermark**  

---

## **1. Load CodeIgniter’s Image Library**  

The `Image Manipulation Library` in CodeIgniter supports **GD2, ImageMagick, and GD**. We will use **GD2** (default).  

---

## **2. Enable Image Manipulation Library**  

Check that **GD2 is installed** on your server. Run:  
```sh
php -i | grep -i gd
```
✅ If GD2 is installed, it will list the GD version.

---

## **3. Upload an Image to Manipulate**  

📁 Create an `uploads` folder in `public/` to store images.  

**Set upload permissions (for Linux users):**  
```sh
chmod 777 writable/uploads
```

---

## **4. Create an Image Controller**  

📁 `app/Controllers/ImageController.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class ImageController extends Controller
{
    public function upload()
    {
        return view('image_upload');
    }

    public function process()
    {
        $file = $this->request->getFile('image');

        if ($file->isValid() && !$file->hasMoved()) {
            $filePath = 'uploads/' . $file->getName();
            $file->move('uploads');

            // Resize Image
            $this->resizeImage($filePath);

            // Crop Image
            $this->cropImage($filePath);

            // Watermark Image
            $this->watermarkImage($filePath);

            return redirect()->to('/image/upload')->with('message', 'Image processed successfully!');
        } else {
            return redirect()->to('/image/upload')->with('error', 'File upload failed.');
        }
    }

    private function resizeImage($path)
    {
        $image = \Config\Services::image()
            ->withFile($path)
            ->resize(300, 300, true, 'auto') // Resize while maintaining aspect ratio
            ->save($path);
    }

    private function cropImage($path)
    {
        $image = \Config\Services::image()
            ->withFile($path)
            ->crop(200, 200, 50, 50) // Crop a 200x200 section starting at (50,50)
            ->save($path);
    }

    private function watermarkImage($path)
    {
        $image = \Config\Services::image()
            ->withFile($path)
            ->text('My Watermark', [
                'color'      => '#ffffff',
                'opacity'    => 0.5,
                'withShadow' => true,
                'hAlign'     => 'center',
                'vAlign'     => 'bottom',
                'fontSize'   => 20
            ])
            ->save($path);
    }
}
```
✅ **This controller:**
1. Uploads the image.
2. Resizes it to **300x300 px** while maintaining aspect ratio.
3. Crops a **200x200 px** section.
4. Adds a **watermark** at the bottom.

---

## **5. Create Image Upload View**  

📁 `app/Views/image_upload.php`  

```php
<!DOCTYPE html>
<html>
<head>
    <title>Upload Image</title>
</head>
<body>
    <h2>Upload an Image</h2>
    <?php if (session()->getFlashdata('message')): ?>
        <p style="color: green;"><?= session()->getFlashdata('message') ?></p>
    <?php endif; ?>
    
    <?php if (session()->getFlashdata('error')): ?>
        <p style="color: red;"><?= session()->getFlashdata('error') ?></p>
    <?php endif; ?>

    <form action="<?= base_url('/image/process') ?>" method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Upload</button>
    </form>
</body>
</html>
```
✅ **This view lets users upload images for processing.**

---

## **6. Define Routes**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/image/upload', 'ImageController::upload');
$routes->post('/image/process', 'ImageController::process');
```
✅ **Now you can access `http://localhost:8080/image/upload` to upload images.**

---

## **7. Testing Image Manipulation**  

1. Start the **CodeIgniter server**:  
   ```sh
   php spark serve
   ```
2. Open your browser and go to:  
   ```
   http://localhost:8080/image/upload
   ```
3. Upload an image and check the `public/uploads/` folder for the processed image.

---

## **8. Summary**  

🚀 **You have learned how to:**  
✅ Upload an image  
✅ Resize an image  
✅ Crop an image  
✅ Add a watermark  

💡 Want to add **thumbnail generation** or **convert images to grayscale**? Let me know! 😊