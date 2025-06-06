Great question, Sandeep! 🔐 When your Laravel app depends on a **private Bitbucket repository**, you need to **authenticate Composer** so it can access that repo during install or CI builds.

---

## ✅ How Do You Authenticate with Private Bitbucket Repositories in `composer.json`?

You have **three secure options**:

---

### 🔐 1. **Use SSH Authentication (Recommended)**

#### ✅ Setup Steps:
- Use an **SSH URL** for the repo:
  
```json
"repositories": [
  {
    "type": "vcs",
    "url": "git@bitbucket.org:your-username/your-private-repo.git"
  }
]
```

- Ensure your **SSH key is added** to:
  - Your **local machine** (`~/.ssh/id_ed25519`)
  - Your **Bitbucket account**:  
    👉 Go to **Bitbucket > Personal Settings > SSH Keys**

- Add the private key to your ssh-agent:
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```

Then run:
```bash
composer require your-vendor/your-private-repo
```

---

### 🔐 2. **Use HTTP Authentication (Access Token or App Password)**

**URL format:**
```json
"repositories": [
  {
    "type": "vcs",
    "url": "https://x-token-auth:{APP_PASSWORD}@bitbucket.org/your-username/your-private-repo.git"
  }
]
```

> Replace `{APP_PASSWORD}` with a **Bitbucket App Password** (not your real password).  
> Generate it from:  
> **Bitbucket > Personal Settings > App passwords**

Set permissions like:
- Repositories: Read
- Projects: Read

---

### ⚙️ 3. **Use Auth Config in `auth.json` (CI-friendly)**

To avoid exposing credentials in `composer.json`, use:

```bash
composer config --global --auth http-basic.bitbucket.org your-username your-app-password
```

Or create a `auth.json` manually in the same dir as `composer.json`:

```json
{
  "http-basic": {
    "bitbucket.org": {
      "username": "your-username",
      "password": "your-app-password"
    }
  }
}
```

Make sure to **exclude this from version control**:

```
.gitignore:
auth.json
```

---

### 🚨 Never Do This:
```json
"repositories": [
  {
    "type": "vcs",
    "url": "https://username:password@bitbucket.org/..."
  }
]
```
> ❌ It's insecure. Prefer `auth.json` or SSH instead.

---
