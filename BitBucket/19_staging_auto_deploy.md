Great deployment-focused question, Sandeep! 🚀  
Here’s **how to automatically deploy your Laravel app to a staging server** (e.g., via SSH) **when pushing to the `develop` branch**, using Bitbucket Pipelines.

---

## ✅ **Auto-Deploy Laravel to Staging Server When Pushing to `develop` (Bitbucket Pipelines)**

---

### 🔧 **1. Setup SSH Access from Bitbucket to Your Staging Server**

#### Step-by-step:
1. Generate SSH key inside your local machine or CI:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "bitbucket-deploy" -f bitbucket_key
   ```
2. Add the **public key** (`bitbucket_key.pub`) to your **staging server’s `~/.ssh/authorized_keys`**
3. Add the **private key** (`bitbucket_key`) to **Bitbucket Pipelines:**

   - Go to:  
     **Repo Settings → Pipelines → SSH Keys → Add SSH Key**
   - Paste your **private key**

---

### 📁 **2. Configure `bitbucket-pipelines.yml` for Auto Deployment on `develop` Push**

```yaml
image: php:8.1

pipelines:
  branches:
    develop:  # Triggers on push to `develop`
      - step:
          name: "🚀 Deploy to Staging Server"
          deployment: staging
          script:
            # Install dependencies
            - apt-get update && apt-get install -y unzip git openssh-client rsync curl
            - curl -sS https://getcomposer.org/installer | php
            - php composer.phar install --no-dev --optimize-autoloader

            # Set up SSH
            - mkdir -p ~/.ssh
            - echo "$KNOWN_HOSTS" > ~/.ssh/known_hosts  # Added as env var or hardcoded with `ssh-keyscan`
            
            # Deploy via rsync (safe, incremental upload)
            - rsync -avz --delete-after ./ user@your-staging-server:/var/www/your-laravel-app

            # Remote commands: migrate + optimize
            - ssh user@your-staging-server "cd /var/www/your-laravel-app && php artisan migrate --force && php artisan config:cache && php artisan route:cache"
```

---

### 🧪 **3. (Optional) Add Known Hosts for SSH Trust**

To avoid prompt during connection:

```bash
ssh-keyscan your-staging-server.com >> ~/.ssh/known_hosts
```

Or, store output as a secure Bitbucket **repository variable** named `KNOWN_HOSTS`:

```bash
KNOWN_HOSTS=$(ssh-keyscan -H your-staging-server.com)
```

---

### 📦 Directory Structure (Server)

Make sure your Laravel app on the staging server has:
```
/var/www/your-laravel-app
  ├── storage/
  ├── vendor/
  └── .env  ← (Managed manually or synced)
```

---

### 🧾 Deployment Best Practices

- Never commit `.env` — use Bitbucket Repository Variables or manage it on the server.
- Use `--no-dev` during Composer install for staging.
- Enable Laravel optimizations: `config:cache`, `route:cache`, `view:cache`.

---

### 📄 Suggested Markdown Filename

```
20_auto_deploy_laravel_staging.md
```

---

Want help writing a full `.env.staging` template, setting up zero-downtime deployment (with Envoy), or syncing assets using GitHub Actions instead of Bitbucket? Just say the word! 💪