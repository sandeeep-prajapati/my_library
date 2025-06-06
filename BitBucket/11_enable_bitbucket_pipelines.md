
---

## ⚙️ **How to Enable and Configure Bitbucket Pipelines for Your Laravel Project**

---

### 🧪 Step 1: Enable Bitbucket Pipelines

1. Go to your repository on Bitbucket  
2. Click on **Pipelines** in the sidebar  
3. Click **Enable Pipelines**

---

### 🛠️ Step 2: Create a `bitbucket-pipelines.yml` File

At your project root (same level as `composer.json`), create:

```yml
# bitbucket-pipelines.yml

image: php:8.1

pipelines:
  default:
    - step:
        name: "Run Laravel Tests"
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
          - php artisan migrate
          - php artisan test
```

---

### ⚙️ Step 3: Add `.env.pipelines`

Create a file called `.env.pipelines` for the pipeline environment:
```env
APP_NAME=Laravel
APP_ENV=testing
APP_KEY=
APP_DEBUG=true
APP_URL=http://localhost

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=homestead
DB_USERNAME=root
DB_PASSWORD=root
```

> Make sure this doesn’t replace your local `.env`!

---

### 🐬 Step 4: Define Database Service

At the bottom of your `bitbucket-pipelines.yml`:
```yml
definitions:
  services:
    mysql:
      image: mysql:5.7
      environment:
        MYSQL_DATABASE: 'homestead'
        MYSQL_ROOT_PASSWORD: 'root'
```

---

### 💡 Step 5: Commit Everything

```bash
git add bitbucket-pipelines.yml .env.pipelines
git commit -m "Add Bitbucket Pipelines for Laravel"
git push origin develop
```

Pipelines will trigger and run Laravel tests automatically on each push!

---

## 🎯 Optional Enhancements

| Feature                     | Add To Config                                  |
|-----------------------------|------------------------------------------------|
| Lint with PHP-CS-Fixer      | `vendor/bin/php-cs-fixer fix --dry-run`       |
| Run Pest tests (if using)   | `./vendor/bin/pest`                           |
| Deploy to production        | Add a `branches:` block & SSH deploy step     |
| Notify on Slack             | Integrate via Pipelines → Settings → Webhooks |

---

## ✅ Final Folder View

```
├── app/
├── routes/
├── .env
├── .env.pipelines ✅
├── bitbucket-pipelines.yml ✅
├── composer.json
└── tests/
```
