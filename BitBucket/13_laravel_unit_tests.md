
---

## ✅ **How to Run `php artisan test` in Bitbucket Pipelines**

### 🗂️ Prerequisites:
- Laravel test cases written (in `tests/Feature/` or `tests/Unit/`)
- `phpunit.xml` file present
- `.env.pipelines` file with testing DB credentials
- Bitbucket Pipelines enabled

---

### 🔧 `bitbucket-pipelines.yml` Example

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: "🔍 Run Laravel PHPUnit Tests"
        services:
          - mysql
        caches:
          - composer
        script:
          - apt-get update && apt-get install -y unzip git zip curl libzip-dev libpng-dev libonig-dev libxml2-dev
          - docker-php-ext-install pdo pdo_mysql zip
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install
          - cp .env.pipelines .env
          - php artisan key:generate
          - php artisan config:cache
          - php artisan migrate --force
          - php artisan test   # ← This runs PHPUnit tests
```

---

### 🐬 MySQL Service Setup

Add this to the bottom of the file:

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

### 📄 Example `.env.pipelines`

```dotenv
APP_ENV=testing
APP_KEY=base64:PLACEHOLDER_KEY
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=homestead
DB_USERNAME=homestead
DB_PASSWORD=secret
```

> 💡 Replace `APP_KEY` with a real one or let Pipelines generate it with `php artisan key:generate`.

---

### 🧪 Sample Output in Pipelines Console

```
PASS  Tests\Feature\HomePageTest
✓ home page loads correctly

PASS  Tests\Unit\UserModelTest
✓ user creation and retrieval works

Tests:  2 passed
Time:   0.59s
```

---

## 💡 Bonus Tips

- Use `--parallel` if you're using Laravel 10+ with parallel testing.
- For larger projects: Split into multiple steps (install, lint, test, deploy).
- Integrate Slack for test result alerts.
- Use Pest PHP if you prefer expressive test syntax.

---
