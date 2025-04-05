Excellent question, Sandeep! 🔐 Managing **Laravel environment variables securely** in Bitbucket is a key DevOps skill—especially when deploying via CI/CD pipelines. Let’s break down **how to do it safely and smartly**.

---

## ✅ **How to Manage Laravel `.env` Variables Securely in Bitbucket**

---

### 🧨 Never Commit `.env` File

Add it to `.gitignore` (Laravel includes this by default):

```bash
.env
```

This prevents secrets (like DB passwords, API keys, etc.) from leaking into your repo.

---

### 🔐 1. **Use Bitbucket Repository Variables**

Bitbucket lets you store **environment variables securely**.

#### ➤ Steps:
1. Go to your Bitbucket repo.
2. Navigate to:  
   **Repository Settings → Pipelines → Repository variables**
3. Add your key-value pairs like:

| Key              | Value             |
|------------------|-------------------|
| `DB_HOST`        | `127.0.0.1`        |
| `DB_DATABASE`    | `homestead`        |
| `DB_USERNAME`    | `homestead`        |
| `DB_PASSWORD`    | `secret`           |
| `APP_KEY`        | `base64:xyz...`    |

✅ These are encrypted and safe—even your teammates can’t see them in plain text.

---

### 🛠️ 2. **Inject Vars in `bitbucket-pipelines.yml`**

Update your pipeline to **dynamically write a `.env` file** from those vars:

```yaml
script:
  - echo "APP_ENV=testing" > .env
  - echo "APP_KEY=$APP_KEY" >> .env
  - echo "DB_CONNECTION=mysql" >> .env
  - echo "DB_HOST=$DB_HOST" >> .env
  - echo "DB_PORT=3306" >> .env
  - echo "DB_DATABASE=$DB_DATABASE" >> .env
  - echo "DB_USERNAME=$DB_USERNAME" >> .env
  - echo "DB_PASSWORD=$DB_PASSWORD" >> .env
```

> 🧠 This generates a secure `.env` file on the fly—without ever pushing it to the repo.

---

### 🗂️ 3. **Alternative: Use `.env.pipelines` (Limited security)**

You can commit a `.env.pipelines` file for **testing only**, but **never for production**.

```dotenv
# .env.pipelines
APP_ENV=testing
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_DATABASE=test_db
DB_USERNAME=test_user
DB_PASSWORD=secret
```

Then:
```bash
cp .env.pipelines .env
```

🚨 Not recommended for real secrets.

---

### 🛡️ 4. **Use Environment-Specific Configuration**

In `config/database.php` or `config/mail.php`, always reference `env('VAR_NAME')` so that Laravel reads from the `.env` file during pipeline builds or local dev.

---

### ✅ Bonus: Use Laravel Envoyer or Bitbucket Deployments for Production

For production, you can:
- Use Laravel Envoyer or Forge to manage `.env` secrets
- Or set Bitbucket **Deployment Environment variables** (`staging`, `production`)

---

### ✅ Sample Output (Console)

```
APP_KEY set.
Environment file created.
Database connection successful.
```

---
