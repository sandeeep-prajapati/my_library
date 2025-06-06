### Laravel 11: Eloquent ORM and Database Modeling

Eloquent ORM (Object-Relational Mapping) is Laravel's built-in ORM that provides a simple and elegant way to interact with your database. It allows you to work with your database using PHP syntax, making it easy to build and manage your database models.

---

### 1. **Eloquent ORM Overview**

- **Definition**: Eloquent is an Active Record implementation that provides a simple way to interact with your database using model classes.
- **Features**:
  - Relationships: Define relationships between different models (one-to-one, one-to-many, many-to-many, etc.).
  - Querying: Build complex queries using a fluent interface.
  - Data Manipulation: Create, read, update, and delete records easily.

---

### 2. **Creating Eloquent Models**

To create a model, you can use the Artisan command:

```bash
php artisan make:model Post
```

This creates a new model file in the `app/Models` directory. By default, Eloquent assumes that the model corresponds to a database table with the plural form of the model name (e.g., `posts` for the `Post` model).

#### Example Model

```php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Post extends Model
{
    protected $fillable = ['title', 'content', 'user_id'];
}
```

- **`$fillable`**: Specifies which attributes are mass assignable.

---

### 3. **Database Migrations**

Migrations are a way to version control your database schema. You can create a migration for your model using the following command:

```bash
php artisan make:migration create_posts_table
```

This creates a new migration file in the `database/migrations` directory. In the migration file, you can define the table structure:

```php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreatePostsTable extends Migration
{
    public function up()
    {
        Schema::create('posts', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->text('content');
            $table->integer('user_id');
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('posts');
    }
}
```

- **Run Migrations**: After creating the migration, run it using:

```bash
php artisan migrate
```

---

### 4. **Basic CRUD Operations with Eloquent**

Eloquent provides a simple interface for performing CRUD operations.

#### Create

```php
$post = Post::create([
    'title' => 'My First Post',
    'content' => 'This is the content of the post.',
    'user_id' => 1,
]);
```

#### Read

- **Get All Records**:

```php
$posts = Post::all();
```

- **Find by ID**:

```php
$post = Post::find(1);
```

#### Update

```php
$post = Post::find(1);
$post->title = 'Updated Title';
$post->save();
```

#### Delete

```php
$post = Post::find(1);
$post->delete();
```

---

### 5. **Defining Relationships**

Eloquent makes it easy to define relationships between models.

#### One-to-Many Relationship

```php
class User extends Model
{
    public function posts()
    {
        return $this->hasMany(Post::class);
    }
}

class Post extends Model
{
    public function user()
    {
        return $this->belongsTo(User::class);
    }
}
```

#### Many-to-Many Relationship

```php
class User extends Model
{
    public function roles()
    {
        return $this->belongsToMany(Role::class);
    }
}

class Role extends Model
{
    public function users()
    {
        return $this->belongsToMany(User::class);
    }
}
```

---

### 6. **Query Scopes**

You can define query scopes in your models for reusable query logic.

#### Example Scope

```php
class Post extends Model
{
    public function scopePublished($query)
    {
        return $query->where('is_published', true);
    }
}

// Usage
$publishedPosts = Post::published()->get();
```

---

### 7. **Accessors and Mutators**

Eloquent allows you to define accessors and mutators to format attributes when retrieving or saving them.

#### Accessor Example

```php
public function getTitleAttribute($value)
{
    return ucwords($value);
}
```

#### Mutator Example

```php
public function setContentAttribute($value)
{
    $this->attributes['content'] = strtolower($value);
}
```

---

### 8. **Soft Deletes**

Eloquent supports soft deletes, which allow you to keep records in the database while marking them as deleted.

```php
use Illuminate\Database\Eloquent\SoftDeletes;

class Post extends Model
{
    use SoftDeletes;

    // Add soft delete column in migration
    $table->softDeletes();
}
```

#### Usage

```php
$post = Post::find(1);
$post->delete(); // Soft delete

$deletedPosts = Post::onlyTrashed()->get(); // Retrieve soft deleted posts
```

---

### Summary

- **Eloquent ORM**: A powerful Active Record implementation for database interaction.
- **Creating Models**: Use `php artisan make:model` to create Eloquent models.
- **Database Migrations**: Version control your database schema with migrations.
- **CRUD Operations**: Perform Create, Read, Update, and Delete operations easily.
- **Relationships**: Define one-to-many and many-to-many relationships between models.
- **Query Scopes**: Create reusable query logic using scopes.
- **Accessors and Mutators**: Format attributes on retrieval and saving.
- **Soft Deletes**: Support for soft deletes allows records to be "deleted" without removing them from the database.

This overview provides a solid foundation for using Eloquent ORM and database modeling in Laravel 11. If you have specific questions or need more examples, feel free to ask!
