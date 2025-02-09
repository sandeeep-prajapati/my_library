# **How to Write Unit Tests for a CodeIgniter Application?**  

Writing unit tests ensures that your CodeIgniter application functions correctly by testing individual components, such as **controllers, models, and libraries**. CodeIgniter uses **PHPUnit** for unit testing.

---

## **1. Install PHPUnit**  
If PHPUnit is not installed, install it using Composer:

```sh
composer require --dev phpunit/phpunit
```
✅ This installs PHPUnit as a development dependency.

---

## **2. Configure PHPUnit for CodeIgniter**  
PHPUnit requires a configuration file.

📁 **Create `phpunit.xml` in your project root:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<phpunit bootstrap="vendor/autoload.php">
    <testsuites>
        <testsuite name="Application Test Suite">
            <directory>./tests</directory>
        </testsuite>
    </testsuites>

    <php>
        <ini name="memory_limit" value="512M"/>
    </php>
</phpunit>
```

✅ This tells PHPUnit to load the `vendor/autoload.php` and run tests in the `tests/` directory.

---

## **3. Create the Tests Directory**  
If it doesn’t exist, create a `tests/` folder in your project root.

📁 **Folder Structure:**
```
tests/
 ├── app/
 │   ├── Models/
 │   │   ├── UserModelTest.php
 │   ├── Controllers/
 │   │   ├── HomeControllerTest.php
 ├── phpunit.xml
```
---

## **4. Write Unit Tests for Models**  
Test database operations in models.

📁 **Example: `tests/app/Models/UserModelTest.php`**
```php
<?php

namespace Tests\App\Models;
use CodeIgniter\Test\CIUnitTestCase;
use App\Models\UserModel;

class UserModelTest extends CIUnitTestCase
{
    public function testFindUserById()
    {
        $model = new UserModel();
        $user = $model->find(1);

        $this->assertNotNull($user, "User should exist");
        $this->assertEquals(1, $user['id']);
    }
}
```
✅ This ensures that the `UserModel` retrieves a user by ID.

---

## **5. Write Unit Tests for Controllers**  
Test if a controller returns the expected response.

📁 **Example: `tests/app/Controllers/HomeControllerTest.php`**
```php
<?php

namespace Tests\App\Controllers;
use CodeIgniter\Test\FeatureTestCase;

class HomeControllerTest extends FeatureTestCase
{
    public function testHomePage()
    {
        $result = $this->get('/');
        $result->assertStatus(200);
        $result->assertSee('Welcome to CodeIgniter');
    }
}
```
✅ This tests if the home page loads successfully.

---

## **6. Run the Tests**  
Run PHPUnit from the project root:

```sh
vendor/bin/phpunit
```
or, if globally installed:

```sh
phpunit
```

✅ Output:
```
PHPUnit 9.5.10 by Sebastian Bergmann and contributors.

..                                                              2 / 2 (100%)

OK (2 tests, 2 assertions)
```

---

## **7. Mocking Database and Services (Advanced)**  
Instead of using a real database, **mock dependencies**.

### **Mocking a Model**
```php
<?php
use CodeIgniter\Test\Mock\MockModel;

class MockUserModelTest extends CIUnitTestCase
{
    public function testMockUserModel()
    {
        $mockModel = $this->createMock(UserModel::class);
        $mockModel->method('find')->willReturn(['id' => 1, 'name' => 'John Doe']);

        $user = $mockModel->find(1);
        $this->assertEquals('John Doe', $user['name']);
    }
}
```

✅ This avoids interacting with a real database.

---

## **Conclusion**  
✅ Install and configure **PHPUnit**.  
✅ Write unit tests for **models and controllers**.  
✅ Use **mocking** to isolate dependencies.  
✅ Run tests using `phpunit` and check the results.  

🚀 Now your CodeIgniter application has **automated tests for reliability and performance!**