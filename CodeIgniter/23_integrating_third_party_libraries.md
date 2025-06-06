## **How to Integrate Third-Party Libraries and APIs in CodeIgniter?**  

Integrating third-party libraries and APIs in CodeIgniter allows you to extend your application's functionality. This guide will cover **installing, configuring, and using external libraries and APIs** in CodeIgniter.  

---

## **1. Using Composer to Install Third-Party Libraries**  
### **Step 1: Install Composer**  
Download and install [Composer](https://getcomposer.org/download/).  

### **Step 2: Initialize Composer in Your Project**  
Navigate to your project directory and run:  
```shell
composer init
```
Then install a library, for example, **GuzzleHTTP** (for API requests):  
```shell
composer require guzzlehttp/guzzle
```
This will download the library inside the `vendor` folder.  

---

## **2. Load Composer Libraries in CodeIgniter**  
### **Enable Composer Autoloading**  
Edit `app/Config/Autoload.php`:  
```php
public $psr4 = [
    'App' => APPPATH,
    'GuzzleHttp' => ROOTPATH . 'vendor/guzzlehttp/guzzle/src'
];
```

Now, you can use Guzzle in your controllers.

### **Example: Fetch Data from an API (GitHub Users API)**  
📁 **`app/Controllers/ApiClient.php`**  
```php
namespace App\Controllers;
use CodeIgniter\Controller;
use GuzzleHttp\Client;

class ApiClient extends Controller {
    public function fetchGitHubUser($username) {
        $client = new Client();
        $response = $client->get("https://api.github.com/users/$username");
        
        echo $response->getBody();
    }
}
```
✅ **Test it in Browser:**  
```
http://localhost:8080/apiclient/fetchGitHubUser/octocat
```

---

## **3. Manually Integrating Third-Party Libraries**  
If a library isn’t available via Composer, you can manually include it.

### **Step 1: Place the Library in `app/Libraries/`**  
Example:  
📁 **`app/Libraries/PdfGenerator.php`**  
```php
namespace App\Libraries;
require_once APPPATH . 'ThirdParty/fpdf/fpdf.php';

class PdfGenerator extends \FPDF {
    public function createPDF($content) {
        $this->AddPage();
        $this->SetFont('Arial', 'B', 16);
        $this->Cell(40, 10, $content);
        $this->Output();
    }
}
```

### **Step 2: Use It in a Controller**  
📁 **`app/Controllers/PdfController.php`**  
```php
namespace App\Controllers;
use App\Libraries\PdfGenerator;

class PdfController extends BaseController {
    public function generate() {
        $pdf = new PdfGenerator();
        $pdf->createPDF("Hello, this is a PDF document!");
    }
}
```
✅ **Test it in Browser:**  
```
http://localhost:8080/pdfcontroller/generate
```

---

## **4. Using API Keys (Example: OpenWeather API)**  
If an API requires an API key, store it in the `.env` file:  
```
OPENWEATHER_API_KEY=your_api_key_here
```

Then, fetch the key and use it:  
```php
$apiKey = getenv('OPENWEATHER_API_KEY');
```

---

## **Conclusion**  
✔ Install third-party libraries via **Composer** or manually.  
✔ Load libraries using **autoload or manual imports**.  
✔ Handle **API keys securely using `.env` files**.  
🚀 Now you can integrate any external service into your CodeIgniter project!