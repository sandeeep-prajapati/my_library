Awesome question, Sandeep! 🔥 Running **Laravel database migrations** using `php artisan migrate` inside Bitbucket Pipelines is essential for preparing your testing environment (especially before running tests). Here's the complete lowdown:

---

## ✅ **How to Run `php artisan migrate` in Bitbucket Pipelines**

---

### 🧠 Why You Need It:

`php artisan migrate` ensures that your **test database schema** is up-to-date **before running tests or seeding data.**

---

### 🗂️ Minimal Setup in `bitbucket-pipelines.yml`

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: "🛠️ Migrate and Test Laravel DB"
        services:
          - mysql
        caches:
          - composer
        script:
          # Install PHP extensions and dependencies
          - apt-get update && apt-get install -y unzip git zip curl libzip-dev libpng-dev libonig-dev libxml2-dev
          - docker-php-ext-install pdo pdo_mysql zip
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install
          
          # Set up environment
          - cp .env.pipelines .env
          - php artisan key:generate
          - php artisan config:cache
          
          # ✅ Run migrations
          - php artisan migrate --force
          
          # ✅ Optional: Run tests
          - php artisan test
```

---

### ⚙️ MySQL Service (at the bottom)

```yaml
definitions:
  services:
    mysql:
      image: mysql:5.7
      environment:
        MYSQL_DATABASE: 'homestead'
        MYSQL_ROOT_PASSWORD: 'root'
        MYSQL_USER: 'homestead'
        MYSQL_PASSWORD: 'secret'
```

---

### 📁 `.env.pipelines` Example

```dotenv
APP_ENV=testing
APP_KEY=base64:PLACEHOLDER
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=homestead
DB_USERNAME=homestead
DB_PASSWORD=secret
```

> 👀 Use `--force` with `php artisan migrate` to skip the confirmation prompt in CI/CD pipelines.

---

### 💡 Pro Tips

| Tip | Why? |
|-----|------|
| `--force` | Required in non-interactive environments like CI |
| Use `php artisan migrate:fresh --seed` | If you need a clean slate with seeders |
| Run `php artisan config:clear` | Sometimes needed if you're caching config or routes |

---

### ✅ Output in Pipelines Console

```
Migrating: 2024_04_01_000001_create_users_table
Migrated:  2024_04_01_000001_create_users_table (0.21 seconds)

PASS  Tests\Feature\LoginTest
✓ login works correctly
```

---
