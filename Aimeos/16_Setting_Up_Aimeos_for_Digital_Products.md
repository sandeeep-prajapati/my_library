Configuring digital products like eBooks, software, and media files for download in Aimeos involves creating a product type that supports digital goods and setting up the necessary logic for managing and delivering these products to customers after purchase.

Here’s a step-by-step guide to configure digital products in Aimeos:

### **1. Create the Digital Product Model**

Aimeos handles products through its `Product` model. To support digital products, you’ll need to create or modify the model to include properties such as the file URL or path where the digital product (eBook, software, media file, etc.) is stored.

#### **A. Extend the Product Model**

If you want to add specific attributes for digital products (like the file location), extend the default `Product` model in Aimeos.

1. **Create a new model for digital products**, for example `DigitalProduct`:

```bash
php artisan make:model Aimeos/Shop/Models/DigitalProduct
```

2. **Extend the product model** and add a field for the digital file URL or path:

```php
namespace Aimeos\Shop\Models;

use Aimeos\Shop\Models\Product;

class DigitalProduct extends Product
{
    // Add custom attributes for digital products
    protected $fillable = ['file_url', 'file_size', 'file_type'];

    // Add custom methods if needed for digital product logic
    public function getDownloadLink()
    {
        return route('product.download', ['id' => $this->id]);
    }
}
```

#### **B. Update the Database Schema**

If you’ve added custom fields (like `file_url` or `file_size`), you'll need to update the database schema to store them.

1. **Create a migration** to add the fields:

```bash
php artisan make:migration add_file_attributes_to_products_table --table=products
```

2. **Modify the migration** to add the necessary fields:

```php
public function up()
{
    Schema::table('products', function (Blueprint $table) {
        $table->string('file_url')->nullable();
        $table->integer('file_size')->nullable(); // Size in KB or MB
        $table->string('file_type')->nullable(); // File type (PDF, ZIP, MP3, etc.)
    });
}

public function down()
{
    Schema::table('products', function (Blueprint $table) {
        $table->dropColumn(['file_url', 'file_size', 'file_type']);
    });
}
```

3. **Run the migration**:

```bash
php artisan migrate
```

---

### **2. Set Up Product Details for Digital Products**

In the backend, you need to ensure that digital product details (such as file URL, size, and type) are properly set when you create or edit a product.

#### **A. Create/Update Digital Products via Admin Panel**

To create or update a digital product:

1. Go to the **Aimeos Admin Panel**.
2. Navigate to **Products**.
3. When creating or editing a product, you'll see the fields for the **file URL**, **file size**, and **file type**. Fill in these fields with the appropriate values.

#### **B. Setting Up the File URL**

For digital products, the `file_url` field should point to the location where the file is stored (e.g., `/storage/products/ebook.pdf`).

- **Example file path:** `public/storage/products/ebook.pdf`
- Ensure that the digital files are stored in the appropriate folder, and ensure the URL matches the location.

---

### **3. Allowing Customers to Download the Digital Product**

Once a customer has purchased a digital product, you need to give them access to download it. This can be done by creating a download link that is associated with the specific product.

#### **A. Create a Route for Product Downloads**

Create a route in `routes/web.php` that will handle the download request for digital products:

```php
Route::get('download/{productId}', [ProductController::class, 'downloadProduct'])->name('product.download');
```

#### **B. Add the Download Logic in the Controller**

You will need to create a controller method that handles the download action. This method will verify that the customer has purchased the product and provide them with the file.

```php
use Aimeos\Shop\Models\DigitalProduct;

class ProductController extends Controller
{
    public function downloadProduct($productId)
    {
        $product = DigitalProduct::find($productId);

        // Check if the product exists and the user is authorized to download it
        if ($product && $this->isAuthorized($product)) {
            // Return the file for download
            return response()->download(storage_path('app/public/' . $product->file_url), $product->name . '.' . $product->file_type);
        }

        // Handle unauthorized or missing product
        return abort(404, 'Product not found or no access');
    }

    private function isAuthorized($product)
    {
        // Implement logic to check if the user has purchased or is authorized to download the product
        // Example: Check if the user has a valid order or purchase record
        return true; // For simplicity, assuming user is authorized
    }
}
```

#### **C. Generate Download Links in the Frontend**

In the product details page, display a download link for digital products. You can add this link in your `resources/views/vendor/aimeos/shop/product/show.blade.php` file:

```blade
@if($product instanceof \Aimeos\Shop\Models\DigitalProduct && $product->file_url)
    <a href="{{ route('product.download', ['productId' => $product->id]) }}" class="btn btn-primary">Download</a>
@endif
```

This will generate a download link for customers who have access to the digital product.

---

### **4. Handling Digital Product Expiration or Limitations**

You might want to add restrictions such as limiting the number of downloads or setting an expiration date for the download link. You can add these features by extending your download logic.

#### **A. Limit Downloads**

You could add a `download_count` field to the product model or create a separate table to track how many times a product has been downloaded.

1. **Add a download count field**:

```php
public function up()
{
    Schema::table('products', function (Blueprint $table) {
        $table->integer('download_count')->default(0);
    });
}
```

2. **Update the download logic to track the number of downloads**:

```php
$product->download_count++;
$product->save();
```

#### **B. Set Expiration Date**

You could also add an expiration date for digital downloads.

1. **Add an expiration date**:

```php
$product->expiration_date = now()->addDays(7);
$product->save();
```

2. **Check for expiration in the download method**:

```php
if ($product->expiration_date && $product->expiration_date < now()) {
    return abort(403, 'The download link has expired');
}
```

---

### **5. Secure File Storage**

To ensure the security of your digital products, it is best to store them outside the public web root and provide controlled access through your application logic.

1. **Store files outside the public folder**:

Move the digital product files to a secure location, such as `storage/app/private/products/`.

2. **Use Laravel's `Storage` facade** to serve files securely:

```php
return response()->download(storage_path('app/private/products/' . $product->file_url));
```

You can also configure Laravel's file storage to use cloud storage (like S3) for added security and scalability.

---

### **6. Testing and Debugging**

Finally, test the entire process to ensure that:

- Digital products are correctly associated with their file paths and URLs.
- The download logic works as expected (e.g., users can download after purchase).
- Download restrictions and expiration policies are enforced.
- All files are stored securely and accessible only to authorized users.

---

### **Conclusion**

By following these steps, you can easily configure digital products like eBooks, software, and media files for download in Aimeos. With the flexibility of Aimeos and Laravel, you can extend product types, manage downloads, enforce restrictions, and ensure a smooth and secure customer experience for digital goods.