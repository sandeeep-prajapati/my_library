### Laravel 11: Database Migrations and Seeding

In Laravel, database migrations and seeding provide a robust way to manage and populate your database schema. Migrations allow you to define your database structure, while seeding allows you to populate your tables with sample data.

---

### 1. **Database Migrations**

#### 1.1. **What are Migrations?**

Migrations are a version control system for your database schema. They allow you to define database tables and columns in a PHP file, making it easy to create, modify, and share the schema with your team.

#### 1.2. **Creating Migrations**

You can create a migration using the Artisan command line tool:

```bash
php artisan make:migration create_users_table
```

This command generates a migration file in the `database/migrations` directory.

#### 1.3. **Defining Migrations**

Open the generated migration file, and you will find two methods: `up()` and `down()`. The `up()` method is used to define the changes to apply to the database, while the `down()` method is used to reverse those changes.

**Example of a Migration:**

```php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateUsersTable extends Migration
{
    public function up()
    {
        Schema::create('users', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->string('email')->unique();
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('users');
    }
}
```

#### 1.4. **Running Migrations**

To run your migrations and create the corresponding tables in your database, use:

```bash
php artisan migrate
```

#### 1.5. **Rolling Back Migrations**

If you need to roll back the last migration, you can use:

```bash
php artisan migrate:rollback
```

To reset all migrations, use:

```bash
php artisan migrate:reset
```

To refresh the migrations (rollback and re-run), use:

```bash
php artisan migrate:refresh
```

---

### 2. **Database Seeding**

#### 2.1. **What is Seeding?**

Seeding allows you to populate your database tables with sample data, which is useful for testing and development purposes.

#### 2.2. **Creating Seeders**

You can create a seeder using the Artisan command:

```bash
php artisan make:seeder UsersTableSeeder
```

This command generates a seeder file in the `database/seeders` directory.

#### 2.3. **Defining Seeders**

Open the generated seeder file and define how you want to populate your table.

**Example of a Seeder:**

```php
namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\User;

class UsersTableSeeder extends Seeder
{
    public function run()
    {
        User::create([
            'name' => 'John Doe',
            'email' => 'john@example.com',
        ]);

        User::create([
            'name' => 'Jane Doe',
            'email' => 'jane@example.com',
        ]);
    }
}
```

#### 2.4. **Running Seeders**

To run your seeders, use the following command:

```bash
php artisan db:seed --class=UsersTableSeeder
```

You can also run all seeders defined in `DatabaseSeeder.php` by using:

```bash
php artisan db:seed
```

#### 2.5. **DatabaseSeeder**

The `DatabaseSeeder.php` file located in the `database/seeders` directory is the main entry point for your seeders. You can call other seeders from this file to organize your seeding process.

**Example of DatabaseSeeder:**

```php
namespace Database\Seeders;

use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    public function run()
    {
        $this->call([
            UsersTableSeeder::class,
            // Other seeders can be called here
        ]);
    }
}
```

---

### 3. **Using Factories with Seeders**

Laravel factories allow you to generate fake data for your models easily. You can create a factory for your User model:

```bash
php artisan make:factory UserFactory --model=User
```

Define the factory in the generated file:

**Example of a Factory:**

```php
namespace Database\Factories;

use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class UserFactory extends Factory
{
    protected $model = User::class;

    public function definition()
    {
        return [
            'name' => $this->faker->name(),
            'email' => $this->faker->unique()->safeEmail(),
        ];
    }
}
```

You can use the factory in your seeder:

```php
public function run()
{
    \App\Models\User::factory()->count(50)->create();
}
```

---

### Summary

- **Migrations**: Version control for your database schema; define tables and columns using PHP.
  - Create migrations using `php artisan make:migration`.
  - Run migrations with `php artisan migrate`.
  - Rollback migrations using `php artisan migrate:rollback`.

- **Seeding**: Populate your database with sample data.
  - Create seeders using `php artisan make:seeder`.
  - Define seeder logic in the generated file.
  - Run seeders with `php artisan db:seed`.

- **Factories**: Generate fake data for testing.
  - Create factories using `php artisan make:factory`.
  - Use factories in seeders for bulk data generation.

This overview provides a solid foundation for using migrations and seeding in Laravel 11. If you have specific questions or need further examples, feel free to ask!