Awesome advanced CI/CD question, Sandeep! 💥  
**Code coverage reports** help ensure your Laravel tests are thoroughly testing your code—and combining this with **Xdebug** in **Bitbucket Pipelines** gives you next-level insight.

---

## ✅ How Do You Generate and Display Code Coverage Reports Using Xdebug in Bitbucket Pipelines?

---

### 🧩 What You'll Need:
- **Xdebug** (PHP extension for coverage)
- **PHPUnit**
- Bitbucket Pipelines with proper config
- Optionally: HTML report artifacts for download/visualization

---

### 📁 Step-by-Step `bitbucket-pipelines.yml` Setup

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: 🧪 PHPUnit with Code Coverage
        artifacts:
          - coverage/**
        caches:
          - composer
        script:
          - apt-get update && apt-get install -y git unzip zip libzip-dev libxml2-dev libpng-dev libonig-dev
          - docker-php-ext-install zip pdo pdo_mysql
          - pecl install xdebug
          - docker-php-ext-enable xdebug
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install
          - cp .env.pipelines .env
          - php artisan key:generate
          - php artisan migrate --force
          - mkdir -p coverage
          - vendor/bin/phpunit --coverage-html coverage
```

---

### 🧪 What This Does:

- Installs **Xdebug**
- Enables code coverage using PHPUnit
- Generates coverage HTML files into a `coverage/` folder
- Saves the coverage folder as an **artifact**, viewable after pipeline run ✅

---

### 📸 View Coverage Report in Bitbucket

After the pipeline runs:

1. Go to the **Pipelines** tab in Bitbucket.
2. Click your latest build.
3. Scroll to **Artifacts** and open the `coverage/` folder.
4. Click `index.html` to view the report locally (Bitbucket doesn’t render HTML inline, but you can download it).

---

### 📂 Optional: PHPUnit XML Configuration (for cleaner setup)

In your `phpunit.xml` file:

```xml
<logging>
  <log type="coverage-html" target="coverage" />
</logging>
```

Then in pipeline:

```bash
vendor/bin/phpunit
```

No need for `--coverage-html` again.

---

### 🔐 Make Sure `.env.pipelines` Includes:

```env
APP_ENV=testing
APP_KEY=base64:...
DB_CONNECTION=sqlite
DB_DATABASE=:memory:
CACHE_DRIVER=array
QUEUE_CONNECTION=sync
```

---
