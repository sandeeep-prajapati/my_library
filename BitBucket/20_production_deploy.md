🔥 Awesome follow-up, Sandeep! Deploying Laravel to a **production server using Bitbucket Pipelines via SCP or FTP** can be a clean and fully automated process — when done right.

---

## ✅ **How to Deploy Laravel to a Production Server via SCP or FTP in Bitbucket Pipelines**

---

### ⚙️ **1. Choose Deployment Method: SCP or FTP?**

| Method | Protocol | Secure? | Recommended |
|--------|----------|---------|-------------|
| SCP    | SSH      | ✅ Yes  | ✅ Best      |
| FTP    | Plain    | ❌ No   | 🚫 Avoid     |
| FTPS/SFTP | Encrypted | ✅ Yes  | ✅ Good |

> 🧠 **Use SCP or SFTP** for production. FTP is not secure and not recommended unless nothing else is available.

---

### 📁 **2. Directory Layout on Server (Assumption)**

Let’s assume your Laravel app is deployed to:

```
/var/www/html/your-laravel-app
```

This should already contain:
- Persistent `.env`
- Writable `storage/` and `bootstrap/cache/`

---

### 🔐 **3. Add SSH Private Key to Bitbucket Pipelines**

1. Generate SSH key pair if you haven’t already:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "bitbucket-prod-deploy" -f bitbucket_prod_key
   ```
2. Add the **public key** (`bitbucket_prod_key.pub`) to:
   ```
   ~/.ssh/authorized_keys
   ```
   on your **production server**.
3. Add the **private key** (`bitbucket_prod_key`) in:
   - **Bitbucket Repo → Settings → Pipelines → SSH Keys → Add SSH Key**

---

### 🚀 **4. Sample `bitbucket-pipelines.yml` File (SCP Deployment)**

```yaml
image: php:8.1

pipelines:
  branches:
    main:  # Deploy only when code is pushed to `main`
      - step:
          name: 🚀 Deploy Laravel to Production via SCP
          deployment: production
          script:
            - apt-get update && apt-get install -y unzip git openssh-client rsync curl
            - curl -sS https://getcomposer.org/installer | php
            - php composer.phar install --no-dev --optimize-autoloader

            # Set known hosts to avoid SSH prompt
            - mkdir -p ~/.ssh
            - echo "$KNOWN_HOSTS" > ~/.ssh/known_hosts

            # Sync code to production server (excluding .env and storage)
            - rsync -avz --delete \
                --exclude '.env' \
                --exclude 'storage/' \
                ./ user@your-server.com:/var/www/html/your-laravel-app

            # Run Laravel post-deploy commands
            - ssh user@your-server.com "cd /var/www/html/your-laravel-app && php artisan migrate --force && php artisan config:cache && php artisan route:cache"
```

---

### 🛡️ **5. For FTP (Not Recommended)**

Use `lftp` if your server only supports FTP:

```yaml
apt-get update && apt-get install -y lftp

lftp -u $FTP_USER,$FTP_PASSWORD -e "mirror -R ./ /your-server-path/; quit" ftp://ftp.your-server.com
```

> ⚠️ Be careful with exposing FTP credentials—**use secured environment variables only**.

---

### 🔒 **6. Manage Secrets via Environment Variables**

| Name           | Value                          |
|----------------|--------------------------------|
| `KNOWN_HOSTS`  | Output of `ssh-keyscan -H ...` |
| `FTP_USER`     | Your FTP username              |
| `FTP_PASSWORD` | Your FTP password              |

---