Great one, Sandeep! 🎯 Running **Laravel Pint** (the official Laravel code style fixer) automatically in your **Bitbucket Pipeline** helps enforce consistent code formatting for your team before merging PRs. Let’s set it up like a boss. 🧼💻

---

## ✅ **How to Run Laravel Pint in Bitbucket Pipelines**

---

### 📦 1. **Install Laravel Pint (locally or globally)**

In your Laravel project:

```bash
composer require laravel/pint --dev
```

---

### 🧠 2. **Add Pint to Your `bitbucket-pipelines.yml`**

Here's an updated pipeline with Pint integrated:

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: "🧽 Laravel Pint & Tests"
        services:
          - mysql
        caches:
          - composer
        script:
          # PHP setup
          - apt-get update && apt-get install -y unzip git zip curl libzip-dev libpng-dev libonig-dev libxml2-dev
          - docker-php-ext-install pdo pdo_mysql zip

          # Install dependencies
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install

          # ✅ Run Laravel Pint
          - ./vendor/bin/pint --test

          # .env & migrate setup
          - cp .env.pipelines .env
          - php artisan key:generate
          - php artisan config:cache
          - php artisan migrate --force

          # ✅ Run tests
          - php artisan test
```

---

### 📄 3. **What `--test` Does**

- `./vendor/bin/pint --test`: **Does not modify files**, only checks if formatting passes.
- Use just `./vendor/bin/pint` to **auto-fix** the code.

---

### 📁 4. Sample `.env.pipelines` (for completeness)

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

---

### 🧪 Output in Pipeline Console

```
Laravel Pint: 0 code style issues found
PASS  Tests\Feature\UserLoginTest
✓ user login works
```

---

### 🔄 Optional: Fail Build on Style Violations

The `--test` flag will automatically **fail the pipeline** if any formatting issue is detected. Great for keeping your team’s code clean!

---

### 🏁 Bonus

- You can also run Pint only on pull requests using:
```yaml
pipelines:
  pull-requests:
    '**':
      - step:
          name: "Pint PR Check"
          script:
            - ./vendor/bin/pint --test
```

---
