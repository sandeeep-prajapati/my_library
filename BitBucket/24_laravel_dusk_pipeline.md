Great question, Sandeep! 💻✨ Running **Laravel Dusk** browser tests in a Bitbucket **CI/CD pipeline** is super powerful for automating frontend testing — but it needs a bit of setup because Dusk relies on a **real browser environment** (Chrome + WebDriver).

---

## ✅ How Do You Configure and Run Laravel Dusk Browser Tests in a CI/CD Pipeline?

---

### 🧩 Requirements:
- Chrome browser
- ChromeDriver
- xvfb (for headless browser display)
- Laravel Dusk installed
- `.env.dusk.ci` environment file

---

### 📁 Sample `bitbucket-pipelines.yml` for Laravel Dusk

```yaml
image: laradock/workspace:latest  # Or use a custom Docker image with Chrome + PHP

pipelines:
  default:
    - step:
        name: 🌐 Run Laravel Dusk Tests
        caches:
          - composer
        script:
          - apt-get update && apt-get install -y unzip curl zip xvfb
          - google-chrome --version || true
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install
          - cp .env.dusk.ci .env
          - php artisan key:generate
          - php artisan migrate --force
          - php artisan serve &  # Serve the Laravel app
          - sleep 5  # Wait for Laravel to boot
          - php artisan dusk
```

---

### 📂 Create `.env.dusk.ci` File

```env
APP_ENV=testing
APP_KEY=base64:...
DB_CONNECTION=sqlite
DB_DATABASE=:memory:
CACHE_DRIVER=array
QUEUE_CONNECTION=sync
APP_URL=http://127.0.0.1:8000
```

---

### 🧪 Install Dusk in Laravel

```bash
composer require --dev laravel/dusk
php artisan dusk:install
```

---

### 💡 Run Dusk with Headless Chrome

Dusk runs headless by default in CI, thanks to:

```php
use Laravel\Dusk\Chrome;

Chrome::useHeadless();
```

You’ll find this in `tests/DuskTestCase.php`.

---

### 🧪 Optional: Save Screenshots on Failures

Dusk automatically saves failure screenshots in:

```
tests/Browser/screenshots/
```

You can store those as **artifacts** in Bitbucket Pipelines:

```yaml
artifacts:
  - tests/Browser/screenshots/**
```

---

### ✅ Checklist:

- [x] Install Chrome & ChromeDriver
- [x] Use `.env.dusk.ci` with proper `APP_URL`
- [x] Serve app before `php artisan dusk`
- [x] Add DB migrations for test environment
- [x] Optionally store screenshots on failure

---
