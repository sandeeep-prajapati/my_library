# **How to Deploy a CodeIgniter Project to a Live Server?**  

Deploying a **CodeIgniter** project to a live server involves several steps, including **configuring the environment, uploading files, setting database connections, and securing the application**.  

---

## **1. Prepare Your Server**  
### ✅ **Choose a Hosting Provider**  
You can deploy CodeIgniter on:  
- **Shared Hosting** (e.g., cPanel, Namecheap, Bluehost)  
- **VPS/Dedicated Server** (e.g., DigitalOcean, AWS, Linode)  
- **Cloud Platforms** (e.g., AWS, Google Cloud, Firebase)  

Ensure that your server meets the requirements:  
✅ **PHP 7.4 or higher**  
✅ **MySQL or MariaDB**  
✅ **Apache/Nginx with mod_rewrite enabled**  

---

## **2. Upload CodeIgniter Files to the Server**  
### ✅ **Using cPanel File Manager (For Shared Hosting)**  
1. **Compress your project** into a `.zip` file.  
2. **Upload it to the `public_html/` folder** using cPanel.  
3. **Extract the ZIP file** inside `public_html`.  

### ✅ **Using FTP (For Any Hosting)**  
1. Install **FileZilla** or use an FTP client.  
2. Connect to your server using FTP credentials.  
3. Upload all project files to the `/public_html` directory.  

### ✅ **Using SSH (For VPS/Cloud Servers)**  
1. **Connect to your server** via SSH:  
   ```sh
   ssh user@your-server-ip
   ```
2. **Clone the project from GitHub (if applicable)**  
   ```sh
   git clone https://github.com/your-repo.git /var/www/html/project-name
   ```
3. **Set proper permissions**  
   ```sh
   chmod -R 755 /var/www/html/project-name
   chmod -R 777 /var/www/html/project-name/writable
   ```

---

## **3. Configure `index.php` and `.htaccess`**  
### ✅ **Remove `index.php` from URLs**  
Modify `.htaccess` in the `public_html/` folder:  

```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    RewriteBase /
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteRule ^(.*)$ index.php/$1 [L]
</IfModule>
```

---

## **4. Configure the `base_url` and Database Settings**  
### ✅ **Update `app/Config/App.php`**  
Set your live **base URL**:  
```php
public $baseURL = 'https://yourdomain.com/';
```

### ✅ **Update `app/Config/Database.php`**  
Edit database credentials:  
```php
public $default = [
    'hostname' => 'localhost',
    'username' => 'your_db_user',
    'password' => 'your_db_password',
    'database' => 'your_db_name',
    'DBDriver' => 'MySQLi',
];
```

---

## **5. Import the Database**  
### ✅ **Using phpMyAdmin**  
1. Open `phpMyAdmin` on your hosting.  
2. Create a **new database** matching `Database.php`.  
3. **Import** the local `.sql` file from your development environment.  

### ✅ **Using MySQL CLI (For VPS)**  
1. Upload your `.sql` file via FTP.  
2. Run:  
   ```sh
   mysql -u your_db_user -p your_db_name < database.sql
   ```

---

## **6. Set File Permissions**  
Ensure that `writable/` has **write permissions**:  
```sh
chmod -R 777 /public_html/writable
```

---

## **7. Configure Environment Variables (Optional)**  
Rename `.env.example` to `.env` and set production mode:  
```ini
CI_ENVIRONMENT = production
```

---

## **8. Optimize Performance**  
### ✅ **Enable Caching** (`app/Config/Cache.php`)  
```php
public $handler = 'file';
public $backupHandler = 'dummy';
```

### ✅ **Minify CSS/JS**  
Use **CDN for assets** like jQuery and Bootstrap.  

---

## **9. Secure Your CodeIgniter Project**  
### ✅ **Disable Debugging (`app/Config/Boot/development.php`)**  
Set:  
```php
error_reporting(0);
ini_set('display_errors', '0');
```

### ✅ **Prevent SQL Injection & XSS**  
- Enable **CSRF protection** in `app/Config/Security.php`:  
  ```php
  public $csrfProtection = true;
  ```

### ✅ **Restrict Direct Access**  
Modify `.htaccess`:  
```apache
# Deny access to important folders
<FilesMatch ".*">
    Order Deny,Allow
    Deny from all
</FilesMatch>
```

---

## **10. Set Up SSL (HTTPS)**  
Most hosting providers offer **free SSL certificates** via Let’s Encrypt.  
Activate HTTPS in `app/Config/App.php`:  
```php
public $baseURL = 'https://yourdomain.com/';
```

---

## **11. Restart Apache/Nginx (For VPS Users)**  
Restart the server:  
```sh
sudo service apache2 restart
```
or  
```sh
sudo systemctl restart nginx
```

---

## **12. Test the Deployment**  
- Open `https://yourdomain.com`  
- Check if **URLs, database, and assets** load correctly.  
- Run:  
  ```sh
  php spark serve
  ```
  If errors occur, check logs in `writable/logs/`.  

---

## **Conclusion**  
🚀 **Your CodeIgniter project is now live!**  
✅ **Configured base URLs and database settings**  
✅ **Uploaded files to the server**  
✅ **Removed `index.php` from URLs**  
✅ **Enabled caching and security features**  

Now, your CodeIgniter application is production-ready! 🎉