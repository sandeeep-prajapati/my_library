Great one, Sandeep! 🔐 The `APP_KEY` is **critical** in Laravel—it encrypts sessions, cookies, and other sensitive data. Mishandling it could expose your app to serious security risks.

Here’s how to handle the `APP_KEY` **safely** and **securely** inside Bitbucket Pipelines:

---

## ✅ **How to Securely Handle Laravel's `APP_KEY` in Bitbucket Pipelines**

---

### 🔐 1. **Generate the `APP_KEY` Locally (Once)**

You only need to generate it once:

```bash
php artisan key:generate --show
```

Example output:
```
base64:s3cr3tEnCrYpt3dK3y==
```

> ✅ **Copy this value.** You’ll store it as a secure variable in Bitbucket.

---

### 🛡️ 2. **Add `APP_KEY` as a Secure Repository Variable**

#### 📍 Steps:
1. Go to your Bitbucket **repo**
2. Navigate to **Repository Settings → Pipelines → Repository Variables**
3. Add:

| Variable Name | Value                        | Secured? |
|---------------|------------------------------|----------|
| `APP_KEY`     | `base64:s3cr3tEnCrYpt3dK3y==` | ✅ Yes   |

> ☑️ Check the **“secured”** box so it’s **encrypted and hidden** in logs.

---

### 🧪 3. **Inject `APP_KEY` into the `.env` in `bitbucket-pipelines.yml`**

Update your `bitbucket-pipelines.yml` file to include:

```yaml
- echo "APP_KEY=$APP_KEY" >> .env
```

✔️ Example:

```yaml
script:
  - echo "APP_NAME=Laravel" > .env
  - echo "APP_ENV=testing" >> .env
  - echo "APP_KEY=$APP_KEY" >> .env
  ...
```

---

### 🚫 **Don’t** Generate a New Key in Pipelines

```bash
# ⚠️ DON'T DO THIS IN PRODUCTION!
php artisan key:generate
```

This regenerates the key every time and will **break encrypted data** (like passwords, tokens, etc.).

---

### 🛠️ Bonus: Use Different Keys for Each Environment

Use Bitbucket’s **Deployment Variables** for:
- `development`
- `staging`
- `production`

This allows each environment to have its own secure `APP_KEY`.

> Go to:  
> **Repo Settings → Deployments → (Choose Environment) → Environment Variables**

---
