# **Defining Custom Routes and Understanding URI Routing in CodeIgniter**  

Routing in CodeIgniter allows you to map **URLs to specific controllers and methods**. By defining custom routes, you can make URLs cleaner, more user-friendly, and better structured for SEO.  

---

## **1. Understanding URI Routing in CodeIgniter**  

By default, CodeIgniter follows the **Controller/Method/Parameter** pattern for handling URLs:  

```
http://yourdomain.com/controller/method/parameter
```

For example, if you have a `UserController` with a `profile()` method, you can access it using:  

```
http://localhost/my_project/user/profile
```

However, using **custom routes**, you can change the way URLs work without modifying controller names.  

---

## **2. Default Routing Configuration**  

### **Location of Route Configuration**  
Routes are defined in the **`app/Config/Routes.php`** file.

### **Default Route**  
CodeIgniter defines a default controller when no specific route is provided.  

```php
$routes->setDefaultController('Home');
$routes->setDefaultMethod('index');
```

This means visiting `http://localhost/my_project/` will execute the `index()` method inside `Home.php`:

```php
class Home extends BaseController
{
    public function index()
    {
        echo "Welcome to CodeIgniter!";
    }
}
```

---

## **3. Defining Custom Routes**  

### **a) Basic Custom Routes**  
You can create custom routes by mapping a specific URI to a controller method:

```php
$routes->get('about', 'Pages::about');
```
Now, visiting:  
```
http://localhost/my_project/about
```
will call the `about()` method inside `Pages.php`:

```php
class Pages extends BaseController
{
    public function about()
    {
        echo "This is the About Us page.";
    }
}
```

### **b) Dynamic Parameters in Routes**  
You can pass parameters using placeholders like `(:num)` for numbers and `(:any)` for any text.

```php
$routes->get('product/(:num)', 'Product::details/$1');
```
- `(:num)`: Matches only numbers  
- `(:any)`: Matches any text  

If a user visits:  
```
http://localhost/my_project/product/5
```
it will call the `details(5)` method in `Product.php`:

```php
class Product extends BaseController
{
    public function details($id)
    {
        echo "Product ID: " . $id;
    }
}
```

---

## **4. Using Named Routes**  

Instead of hardcoding URLs, you can create named routes:

```php
$routes->get('contact', 'Pages::contact', ['as' => 'contactPage']);
```

Now, you can generate URLs dynamically in views:

```php
<a href="<?= route_to('contactPage') ?>">Contact Us</a>
```

---

## **5. Handling HTTP Methods in Routes**  

You can define routes for different HTTP methods like `POST`, `PUT`, `DELETE`, etc.

```php
$routes->post('form/submit', 'FormController::submit');
$routes->put('profile/update', 'UserController::updateProfile');
$routes->delete('user/delete/(:num)', 'UserController::delete/$1');
```

Now, only **POST requests** can access the `submit()` method.

---

## **6. Routing with Closures (Anonymous Functions)**  

Instead of using a controller, you can define simple logic inside routes:

```php
$routes->get('hello', function () {
    return "Hello, World!";
});
```

Visiting `http://localhost/my_project/hello` will display:  

```
Hello, World!
```

---

## **7. Creating a Custom 404 Page**  

By default, CodeIgniter shows a generic 404 error when a route is not found. You can customize it by defining a **fallback route**:

```php
$routes->set404Override(function () {
    echo "Oops! The page you are looking for does not exist.";
});
```

You can also redirect to a custom view:

```php
$routes->set404Override('ErrorController::show404');
```

Then, create `ErrorController.php`:

```php
class ErrorController extends BaseController
{
    public function show404()
    {
        return view('errors/custom_404');
    }
}
```

---

## **8. Enabling/Disabling Auto Routing**  

Auto-routing allows you to call controllers and methods dynamically based on the URL structure.  

**Enable Auto Routing (For Development Only)**  

In `app/Config/Routes.php`, add:

```php
$routes->setAutoRoute(true);
```

🚨 **Warning:** Auto-routing is insecure in production because it allows unintended access to controllers.

To **disable** it for security, set:

```php
$routes->setAutoRoute(false);
```

---

## **9. Testing Routes**  

After defining routes, you can check them using the CLI command:

```sh
php spark routes
```

This will display all registered routes.

---

## **Conclusion**  

✔ **Basic Routing** – Map URIs to controllers.  
✔ **Dynamic Routes** – Use placeholders for variables.  
✔ **Named Routes** – Create user-friendly links.  
✔ **HTTP Method Routing** – Restrict routes to GET, POST, PUT, DELETE.  
✔ **Custom 404 Pages** – Improve user experience.  
