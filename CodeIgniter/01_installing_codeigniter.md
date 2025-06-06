### **Installing CodeIgniter and Setting Up a New Project**  

CodeIgniter is a lightweight PHP framework that follows the MVC (Model-View-Controller) architecture. Here’s how you can install it and set up your first project.  

---

## **1. System Requirements**  
Before installing CodeIgniter, ensure your system meets the following requirements:  
- PHP version **7.4 or higher**  
- MySQL or MariaDB (Optional for database-driven applications)  
- Apache or Nginx web server (Recommended)  
- Composer (Optional for dependency management)  

---

## **2. Installation Methods**  

### **Method 1: Installing via Composer (Recommended for CI4)**
CodeIgniter 4 supports installation using Composer.  

1. **Open Terminal or Command Prompt** and navigate to the directory where you want to install the project.  
2. Run the following command:  
   ```sh
   composer create-project codeigniter4/appstarter my_project
   ```
   Replace `my_project` with your desired project name.  

3. Navigate into the project folder:  
   ```sh
   cd my_project
   ```
4. Start the built-in PHP development server:  
   ```sh
   php spark serve
   ```
   This will start the application at `http://localhost:8080`.  

---

### **Method 2: Installing Manually (ZIP Download)**
If you don’t want to use Composer, you can install CodeIgniter manually.  

1. **Download CodeIgniter** from the official website:  
   [https://codeigniter.com/download](https://codeigniter.com/download)  
2. **Extract the ZIP file** into your web server’s root directory (`htdocs` for XAMPP or `/var/www/html/` for Linux).  
3. **Rename the folder** to your project name, e.g., `my_project`.  
4. **Set the Base URL** in `app/Config/App.php`:  
   ```php
   public $baseURL = 'http://localhost/my_project/';
   ```
5. **Run the application** using a local server (XAMPP, WAMP, or LAMP).  

---

## **3. Project Structure Overview (CodeIgniter 4)**
After installation, your project will have the following structure:  

```
my_project/
│── app/             # Contains controllers, models, views, and config files
│── public/          # Publicly accessible files (index.php, assets, etc.)
│── system/          # Core framework files (Do not modify)
│── writable/        # Cache, logs, and temporary data
│── tests/           # Testing files
│── env              # Environment configuration file
│── composer.json    # Composer dependencies
│── spark            # CLI tool for development
```

---

## **4. Running Your First CodeIgniter App**
- If using Composer:  
  ```sh
  php spark serve
  ```
- If using Apache/XAMPP:  
  - Open `http://localhost/my_project/public` in your browser.  

---

## **5. Troubleshooting Common Issues**  
- **Problem:** Page not found error  
  **Solution:** Ensure the `public` folder is the document root or set up `.htaccess` to remove `public` from the URL.  
- **Problem:** Missing `intl` PHP extension  
  **Solution:** Enable `extension=intl` in your `php.ini` file.  

---

## **Next Steps**  
Now that CodeIgniter is installed, you can:  
- Learn about **Controllers, Models, and Views**.  
- Set up **routing** for clean URLs.  
- Connect CodeIgniter to a **database**.  
