### 🔐 How Can You Publish a **Private NPM Package** for Internal Use Within an Organization?

---

If you want to share an NPM package **only within your team, company, or organization**, you can publish it as a **private package**. This ensures your code stays internal — not visible on the public npm registry.

Let’s walk through how to **create, publish, and install** a private NPM package step by step 👇

---

## ✅ 1. Use a **Scoped Package Name**

All private packages **must be scoped** (e.g., `@org-name/package-name`).

Update your `package.json`:

```json
{
  "name": "@myorg/internal-lib",
  "version": "1.0.0",
  "main": "dist/index.js",
  "license": "UNLICENSED",  // or MIT if needed
  "private": false          // must be false for publishing
}
```

> 🔐 The `"private": true` field prevents any publishing — only use it for monorepos.

---

## ✅ 2. Log in to NPM

```bash
npm login
```

Use a **paid npm account** (Pro or Team plan) or use **GitHub Packages** if your organization uses GitHub.

---

## ✅ 3. Publish as Private

```bash
npm publish --access restricted
```

> `--access restricted` ensures the package is **private** even though you publish to the public registry.

---

## 🏗️ 4. Set Up Access Control

### 🔧 On [npmjs.com](https://npmjs.com):

* Go to the package’s page.
* Click **Access** → Add team members or collaborators.
* Set roles (Read, Write, Admin).

> Or use **organizations** to manage access more easily for multiple private packages.

---

## 🧪 5. Installing a Private Package

On another project (with access permissions), run:

```bash
npm install @myorg/internal-lib
```

🔒 The developer must be **logged into NPM** using an account that has access to the private package.

---

## 📦 Optional: Use `.npmrc` to Manage Authentication

Add this to the consuming project’s `.npmrc`:

```ini
//registry.npmjs.org/:_authToken=YOUR_NPM_TOKEN
```

Or for scoped packages only:

```ini
@myorg:registry=https://registry.npmjs.org/
//registry.npmjs.org/:_authToken=YOUR_NPM_TOKEN
```

Generate a token from:
👉 [https://www.npmjs.com/settings/YOUR\_USERNAME/tokens](https://www.npmjs.com/settings/YOUR_USERNAME/tokens)

---

## 🧱 Alternatives: Use a Private Registry

If you don’t want to use the public NPM registry at all:

### ✅ Use Verdaccio (self-hosted)

* Install: `npm install -g verdaccio`
* Run: `verdaccio`
* Publish and install using your own registry.

### ✅ Use GitHub Packages

For GitHub:

```json
"publishConfig": {
  "registry": "https://npm.pkg.github.com/"
}
```

Then:

```bash
npm publish
```

---

## 🚨 Important Notes

| Rule                           | Why It Matters                                 |
| ------------------------------ | ---------------------------------------------- |
| Must be scoped                 | Private packages require scoped names          |
| Must use `--access restricted` | Ensures the package is not visible publicly    |
| Requires paid plan             | Free accounts can't publish private packages   |
| Use tokens for CI/CD           | Tokens are safer than storing user credentials |

---

## ✅ Summary

| Step               | Command/Action                    |
| ------------------ | --------------------------------- |
| Scope the name     | `"name": "@org/pkg-name"`         |
| Build the package  | `npm run build`                   |
| Publish privately  | `npm publish --access restricted` |
| Install in project | `npm install @org/pkg-name`       |
| Manage access      | Invite users via npm dashboard    |
| Use `.npmrc`       | For tokens and automation         |

---

> 🧠 **In short**:
> Use `--access restricted` with scoped package names and manage permissions through the NPM dashboard or org settings. For full control, consider hosting your own private registry.

Would you like a **`.npmrc` and CI/CD-friendly boilerplate** to publish private packages automatically from GitHub Actions?
