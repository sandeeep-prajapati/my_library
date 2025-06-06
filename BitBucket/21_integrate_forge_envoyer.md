Great one, Sandeep! 🚀  
Integrating **Bitbucket Pipelines with Laravel Forge or Envoyer** allows you to automate **zero-downtime deployments** with robust Laravel hosting tools. Here’s how you do it smoothly:

---

## ✅ How to Integrate Bitbucket Pipelines with Laravel Forge or Envoyer for Seamless Deployment

---

### 🚀 Option 1: **Bitbucket Pipelines + Laravel Forge**

Laravel Forge provides a **webhook-based deployment** system. You simply trigger a deployment via a **POST request**.

#### 🔧 Steps:

1. **Find your Forge deployment webhook**  
   Go to **Forge > Your Server > Site > Deployments > Deployment Script**  
   You'll see a **Deploy Webhook URL**, e.g.:
   ```
   https://forge.laravel.com/servers/123456/sites/987654/deploy/http?token=abc123
   ```

2. **Add this webhook URL to Bitbucket Pipelines** as a **repository variable**:
   - Name: `FORGE_DEPLOY_HOOK`
   - Value: `https://forge.laravel.com/...`

3. **Edit `bitbucket-pipelines.yml` to call the webhook:**

```yaml
image: php:8.1

pipelines:
  branches:
    main:
      - step:
          name: 🚀 Trigger Laravel Forge Deploy
          deployment: production
          script:
            - apt-get update && apt-get install -y curl
            - curl -X POST "$FORGE_DEPLOY_HOOK"
```

✅ That’s it! Forge will run your deployment script when a push to `main` occurs.

---

### 🌐 Option 2: **Bitbucket Pipelines + Laravel Envoyer**

Laravel [Envoyer](https://envoyer.io) is made for **zero-downtime deployment**. It works similarly via **webhook triggers**.

#### 🔧 Steps:

1. **Get your Envoyer deploy hook:**
   - Go to your Envoyer project → **Deployments**
   - Copy the **Deployment Hook URL**, e.g.:
     ```
     https://envoyer.io/deploy/your-app-id/your-token
     ```

2. **Add it to Bitbucket repo as an env variable:**
   - Name: `ENVOYER_DEPLOY_HOOK`
   - Value: `https://envoyer.io/deploy/...`

3. **Trigger it from Bitbucket Pipelines like this:**

```yaml
image: php:8.1

pipelines:
  branches:
    main:
      - step:
          name: 🚀 Trigger Laravel Envoyer Deploy
          deployment: production
          script:
            - apt-get update && apt-get install -y curl
            - curl -X POST "$ENVOYER_DEPLOY_HOOK"
```

✅ Once triggered, Envoyer handles SSH, caching, migrations, symlinks — all with zero-downtime magic ✨

---

### 💡 Pro Tips

- **Don’t expose webhook URLs in code** – always use **Bitbucket Repository Variables**.
- You can **chain tests + build + deploy** for production reliability.
- Want to run extra steps before webhook call? Just insert before the `curl` command:
  ```bash
  php artisan test
  php artisan pint
  ```

---