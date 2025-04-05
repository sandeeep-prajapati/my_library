Great question, Sandeep! 🎯 Here's a clean and well-structured **basic `bitbucket-pipelines.yml` file** tailored for a Laravel project — perfect for running your tests and setting up CI/CD with **minimal configuration**.

---

## ✅ **Basic `bitbucket-pipelines.yml` for Laravel**

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: "🔧 Install Dependencies & Run Tests"
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

## 🔧 **MySQL Service Configuration**

At the bottom of the file, define MySQL settings:

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

## 📁 `.env.pipelines` (Example)

```dotenv
APP_NAME=Laravel
APP_ENV=testing
APP_KEY=
APP_DEBUG=true
APP_URL=http://localhost

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=homestead
DB_USERNAME=homestead
DB_PASSWORD=secret
```

---

## 🧪 Runs Automatically On:

- Pushes to branches (`develop`, `main`, etc.)
- Pull requests (if configured)
- Optional: manual triggers

---

## 🔄 Next Steps

- ✅ Add caching for `node_modules` if using Laravel Mix
- ✅ Add deployment steps to staging or production
- ✅ Add PHP-CS-Fixer or Pest for clean code and unit testing

---
