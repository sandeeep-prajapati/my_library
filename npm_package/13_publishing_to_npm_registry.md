### 🚀 How Do You Publish Your Package to the Public NPM Registry?

Publishing your package to [npmjs.com](https://www.npmjs.com) makes it available for millions of developers to install and use. Here's a **step-by-step guide** to publish your first NPM package correctly and safely.

---

## ✅ Step 1: Prepare Your Package Structure

Your project should look something like this:

```
my-package/
├── src/
│   └── index.ts
├── dist/
│   └── index.js
├── package.json
├── README.md
├── LICENSE
├── .gitignore
├── tsconfig.json
```

---

## 📦 Step 2: Create a `package.json`

If you haven't already:

```bash
npm init
```

Or the quick version:

```bash
npm init -y
```

Then update the key fields:

```json
{
  "name": "@your-username/my-utils",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "license": "MIT",
  "files": ["dist", "README.md"],
  "scripts": {
    "build": "tsc"
  }
}
```

> ✅ Tip: Use **scoped packages** like `@sandeep/my-utils` to avoid name conflicts.

---

## 🔑 Step 3: Log in to NPM

If you haven't logged in before:

```bash
npm login
```

It will ask for your:

* Username
* Password
* Email address

If you don’t have an account yet:
👉 Create one at [https://www.npmjs.com/signup](https://www.npmjs.com/signup)

---

## 🛠 Step 4: Build Your Package

If you're using TypeScript, run:

```bash
npm run build
```

This should generate your compiled code (e.g., in `dist/`).

---

## 🔎 Step 5: Verify What Will Be Published

Check which files will be included:

```bash
npm pack --dry-run
```

This shows the contents that will be published. Make sure:

* Only necessary files are included
* You’ve excluded `src/`, `test/`, `node_modules/`, etc. using `.npmignore` or `files` field

---

## 🚀 Step 6: Publish to NPM

```bash
npm publish --access public
```

> Use `--access public` **only for scoped packages** (like `@sandeep/my-utils`). Unscoped packages are public by default.

---

## 🎉 Step 7: Confirm the Publish

Visit:

```
https://www.npmjs.com/package/@your-username/your-package-name
```

✅ You’ll see your README, version, and installation instructions.

---

## 🆕 Step 8: Updating Later

After making changes, **bump the version** using semantic versioning:

```bash
npm version patch    # 1.0.0 → 1.0.1
npm version minor    # 1.0.1 → 1.1.0
npm version major    # 1.1.0 → 2.0.0
```

Then:

```bash
npm publish
```

---

## ❌ Common Errors

| Error                                   | Fix                                                                 |
| --------------------------------------- | ------------------------------------------------------------------- |
| `You do not have permission to publish` | Make sure your package name is scoped to your username or unclaimed |
| `Package already exists`                | Bump the version                                                    |
| `403 Forbidden`                         | Use `--access public` for scoped packages                           |

---

## 📝 Summary Checklist

✅ You have:

* `package.json` with `name`, `version`, `main`, `types`
* Compiled files in `dist/`
* `.npmignore` or `files` field
* A valid `README.md` and `LICENSE`
* Logged into NPM
* Published with `npm publish --access public`

---

> 🧠 **In short**: Build your code, check your metadata, log in, and run `npm publish`. Semantic versioning and a clean `README.md` help make your package professional and trusted.

Would you like a ready-to-publish starter template with build + publish scripts?
