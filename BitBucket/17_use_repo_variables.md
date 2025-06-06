Awesome follow-up, Sandeep! 💥 Here's exactly how you **configure Bitbucket Repository Variables** and use them to **inject a secure `.env` file during your Laravel pipeline builds**.

---

## ✅ **How to Configure Bitbucket Repository Variables for Laravel `.env` Injection**

---

### 🛡️ **1. Define Variables in Bitbucket**

#### 📌 Steps:
1. Go to your **Bitbucket repository**
2. Click on **⚙️ Repository settings** → **Pipelines** → **Repository variables**
3. Add key-value pairs for your `.env` variables like:

| **Variable Name**  | **Value**                | **Secured** |
|--------------------|--------------------------|-------------|
| `APP_KEY`          | `base64:xyz...`          | ✅          |
| `APP_ENV`          | `testing`                | ❌          |
| `DB_CONNECTION`    | `mysql`                  | ❌          |
| `DB_HOST`          | `127.0.0.1`              | ❌          |
| `DB_PORT`          | `3306`                   | ❌          |
| `DB_DATABASE`      | `homestead`              | ❌          |
| `DB_USERNAME`      | `homestead`              | ❌          |
| `DB_PASSWORD`      | `secret123`              | ✅          |

> 💡 Secure variables are encrypted and hidden in logs.

---

### 🧪 **2. Inject Variables into `.env` in `bitbucket-pipelines.yml`**

You can generate the `.env` file during the pipeline build using `echo` commands:

```yaml
image: php:8.1

pipelines:
  default:
    - step:
        name: "🚀 Build & Test Laravel App"
        services:
          - mysql
        caches:
          - composer
        script:
          - apt-get update && apt-get install -y unzip git curl libzip-dev libpng-dev libonig-dev libxml2-dev
          - docker-php-ext-install pdo pdo_mysql zip

          # Install Composer
          - curl -sS https://getcomposer.org/installer | php
          - php composer.phar install

          # ✅ Create .env file from Bitbucket variables
          - echo "APP_NAME=Laravel" > .env
          - echo "APP_ENV=$APP_ENV" >> .env
          - echo "APP_KEY=$APP_KEY" >> .env
          - echo "APP_DEBUG=true" >> .env
          - echo "APP_URL=http://localhost" >> .env
          - echo "DB_CONNECTION=$DB_CONNECTION" >> .env
          - echo "DB_HOST=$DB_HOST" >> .env
          - echo "DB_PORT=$DB_PORT" >> .env
          - echo "DB_DATABASE=$DB_DATABASE" >> .env
          - echo "DB_USERNAME=$DB_USERNAME" >> .env
          - echo "DB_PASSWORD=$DB_PASSWORD" >> .env

          # Laravel setup
          - php artisan key:generate
          - php artisan migrate --force
          - php artisan test
```

---

### 🧾 Output (in pipeline logs):

```
.env file created with environment variables.
Application key set successfully.
Database migration successful.
Running tests...
```

---

### ⚙️ Services (optional)

Add MySQL or Redis as needed:

```yaml
definitions:
  services:
    mysql:
      image: mysql:5.7
      environment:
        MYSQL_ROOT_PASSWORD: root
        MYSQL_DATABASE: homestead
        MYSQL_USER: homestead
        MYSQL_PASSWORD: secret123
```

---
