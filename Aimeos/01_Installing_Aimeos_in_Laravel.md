### **How to Install Aimeos in a Laravel Project Using Composer**

Aimeos is a high-performance, open-source eCommerce framework for Laravel. Follow these steps to install Aimeos in your Laravel project using Composer.

---

#### **Step 1: Install Laravel**
Ensure you have Laravel installed. If not, create a new Laravel project:

```bash
composer create-project --prefer-dist laravel/laravel my-laravel-store
cd my-laravel-store
```

---

#### **Step 2: Install Aimeos Package**
Run the following Composer command to install the Aimeos Laravel package:

```bash
composer require aimeos/aimeos-laravel:~2024.04
```

---

#### **Step 3: Publish Aimeos Resources**
Publish the necessary configuration files and assets by running:

```bash
php artisan vendor:publish --tag=aimeos
```

This will create configuration files like `config/shop.php` and other required assets.

---

#### **Step 4: Run Database Migrations**
Migrate the Aimeos database tables to your configured database:

```bash
php artisan migrate
```

---

#### **Step 5: Create Default Admin Account**
To create an admin account for the Aimeos backend, run:

```bash
php artisan aimeos:account --super admin admin@example.com password
```

Replace `admin@example.com` with your desired email and `password` with a secure password.

---

#### **Step 6: Seed the Database**
Optionally, you can seed your database with sample data:

```bash
php artisan aimeos:setup
```

---

#### **Step 7: Serve the Application**
Start the development server to test your installation:

```bash
php artisan serve
```

Open your browser and visit:  
- **Frontend:** `http://localhost:8000/shop`
- **Admin Panel:** `http://localhost:8000/admin`

Use the admin credentials created in Step 5 to log in to the admin panel.

---

#### **Tips**
- Make sure your `.env` file is configured for your database connection.
- Install and enable required PHP extensions, such as `mbstring`, `openssl`, `pdo`, and `json`.

---

That's it! Aimeos is now successfully installed in your Laravel project. You can start configuring it and building your eCommerce store.