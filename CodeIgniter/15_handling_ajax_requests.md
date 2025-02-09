# **How to Handle AJAX Requests Using jQuery and CodeIgniter?**  

AJAX (Asynchronous JavaScript and XML) allows web applications to send and receive data from a server without reloading the page. CodeIgniter makes it easy to handle AJAX requests using jQuery and its lightweight controllers.  

---

## **1. Setting Up the Project**  

Ensure jQuery is included in your project:  

📁 `app/Views/ajax_view.php`  

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AJAX in CodeIgniter</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h2>AJAX Example in CodeIgniter</h2>

    <input type="text" id="name" placeholder="Enter your name">
    <button id="submit">Send</button>

    <h3>Response:</h3>
    <p id="response"></p>

    <script>
        $(document).ready(function(){
            $("#submit").click(function(){
                var name = $("#name").val();

                $.ajax({
                    url: "<?= base_url('ajax/submitData') ?>",
                    type: "POST",
                    data: { name: name },
                    dataType: "json",
                    success: function(response) {
                        $("#response").text("Server Response: " + response.message);
                    },
                    error: function() {
                        $("#response").text("Error processing request.");
                    }
                });
            });
        });
    </script>
</body>
</html>
```
✅ **This sends a name to the server using AJAX and displays the response.**  

---

## **2. Creating the AJAX Controller**  

📁 `app/Controllers/AjaxController.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;
use CodeIgniter\HTTP\ResponseInterface;

class AjaxController extends Controller
{
    public function index()
    {
        return view('ajax_view');
    }

    public function submitData()
    {
        $name = $this->request->getPost('name');

        if ($name) {
            return $this->response->setJSON(['message' => "Hello, $name!"]);
        } else {
            return $this->response->setStatusCode(ResponseInterface::HTTP_BAD_REQUEST)
                                  ->setJSON(['message' => 'Name is required!']);
        }
    }
}
```
✅ **Handles AJAX requests and returns a JSON response.**  

---

## **3. Defining Routes for AJAX**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/ajax', 'AjaxController::index');
$routes->post('/ajax/submitData', 'AjaxController::submitData');
```
✅ **Sets up the route for AJAX requests.**  

---

## **4. Testing the AJAX Request**  

1. Open **`http://localhost:8080/ajax`** in your browser.  
2. Enter a name and click the "Send" button.  
3. The page updates **without reloading** with a response like:  
   ```
   Server Response: Hello, John!
   ```

---

## **5. Handling Errors and Debugging**  

If you encounter errors:  
✅ **Check Network Requests in Browser Developer Tools (F12 → Network → XHR).**  
✅ **Use `log_message('error', $errorMessage);` to debug in `writable/logs/`.**  

---

## **Next Steps**  

🚀 **Want to send more data?** Expand the AJAX request.  
🚀 **Need database interaction?** Use a model to store submitted data.  

Would you like an example with **AJAX + Database Integration**? 😊