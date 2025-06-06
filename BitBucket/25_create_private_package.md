Perfect question, Sandeep! 🌟 Hosting and managing a **Laravel package** in a **private Bitbucket repository** is a great way to modularize your codebase and reuse logic across projects securely.

---

## ✅ How Do You Create and Host a Laravel Package in a Private Bitbucket Repository?

---

### 📦 Step 1: Create Your Laravel Package

Your package directory structure should look like this:

```
packages/
└── YourVendor/
    └── YourPackage/
        ├── src/
        │   └── YourServiceProvider.php
        ├── composer.json
        └── README.md
```

Example `composer.json`:

```json
{
  "name": "your-vendor/your-package",
  "description": "Custom Laravel package",
  "autoload": {
    "psr-4": {
      "YourVendor\\YourPackage\\": "src/"
    }
  },
  "extra": {
    "laravel": {
      "providers": [
        "YourVendor\\YourPackage\\YourServiceProvider"
      ]
    }
  }
}
```

---

### 🔐 Step 2: Push Package to Bitbucket (Private Repo)

1. Initialize Git in the package folder:
   ```bash
   git init
   git remote add origin git@bitbucket.org:your-username/your-package.git
   git add .
   git commit -m "Initial commit"
   git push -u origin master
   ```

---

### 📥 Step 3: Require Private Package via Composer

In your Laravel app’s `composer.json`:

```json
"repositories": [
  {
    "type": "vcs",
    "url": "git@bitbucket.org:your-username/your-package.git"
  }
],
"require": {
  "your-vendor/your-package": "*"
}
```

Then run:

```bash
composer require your-vendor/your-package
```

---

### 🔑 Step 4: Set Up SSH Access for Composer (Recommended)

Ensure your machine has SSH access to Bitbucket:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
ssh-add ~/.ssh/id_ed25519
```

Add the **public key** to Bitbucket:
> Bitbucket → Personal Settings → SSH Keys → Add Key

---

### 💡 Pro Tips

- If you're using **Bitbucket Pipelines**, make sure to add the same SSH key to Pipelines' SSH settings.
- Structure the package for **PSR-4 autoloading** and **service provider discovery**.
- Tag your releases with Git:
  ```bash
  git tag v1.0.0
  git push origin v1.0.0
  ```

---
