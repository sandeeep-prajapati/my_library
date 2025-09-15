### ❗ What Are the Most Common Errors While Publishing NPM Packages — and How to Fix Them?

---

Publishing an NPM package can throw unexpected errors — especially the first few times. Here’s a list of the **most common issues**, along with their causes and fixes, so you can publish with confidence. 🚀

---

## 🧨 1. `403 Forbidden - You do not have permission to publish`

### 🔎 Cause:

* You’re trying to publish a **package name** that already exists and **you don’t own it**.
* Or you’re trying to publish under a **scope** you don’t control.

### ✅ Fix:

* Choose a unique package name or use a **scoped package** like `@yourname/your-package`.
* Run:

```bash
npm whoami
```

To confirm you're logged in with the correct account.

---

## 📦 2. `403 Forbidden - You cannot publish over the previously published versions`

### 🔎 Cause:

You're trying to **republish a version number** that has already been published.

### ✅ Fix:

* **Bump the version** using Semantic Versioning:

```bash
npm version patch
# or
npm version minor
# or
npm version major
```

Then re-run:

```bash
npm publish
```

---

## ⚠️ 3. `You must sign up for private packages`

### 🔎 Cause:

You’re publishing a **scoped package** without setting it to public.

### ✅ Fix:

Publish using the `--access public` flag:

```bash
npm publish --access public
```

> Scoped packages (e.g., `@sandeep/my-utils`) are **private by default**.

---

## ❌ 4. `No README data` on npmjs.com

### 🔎 Cause:

You didn’t include a `README.md` file, or it's empty.

### ✅ Fix:

* Add a meaningful `README.md` to the root of your project.
* Ensure it’s added in your `package.json` under:

```json
"files": ["dist", "README.md"]
```

---

## 🗃️ 5. `package.json` is missing `name` or `version`

### 🔎 Cause:

Your `package.json` doesn’t follow NPM's required format.

### ✅ Fix:

Ensure `package.json` includes these keys:

```json
{
  "name": "@sandeep/my-utils",
  "version": "1.0.0",
  "main": "dist/index.js",
  "license": "MIT"
}
```

You can regenerate it with:

```bash
npm init
```

---

## 📁 6. Publishing unwanted files (e.g., `node_modules`, `src`)

### 🔎 Cause:

You forgot to configure `.npmignore` or `files` in `package.json`.

### ✅ Fix:

Option 1: Use a `.npmignore` file:

```
src/
node_modules/
test/
tsconfig.json
rollup.config.js
```

Option 2: Use the `files` field in `package.json`:

```json
"files": ["dist", "README.md"]
```

Use this to **whitelist** what you want published.

---

## 🔒 7. Forgot to log in or logged in with the wrong account

### 🔎 Cause:

You haven’t logged in or you're logged in with the wrong NPM account.

### ✅ Fix:

```bash
npm logout
npm login
```

Make sure you’re logged in with the correct user (check with `npm whoami`).

---

## 🕒 8. Cannot unpublish after 72 hours

### 🔎 Cause:

NPM **restricts unpublishing** a package **after 72 hours** to avoid breaking others’ projects.

### ✅ Fix:

* Use `npm deprecate` instead:

```bash
npm deprecate @sandeep/my-utils@"<2.0.0" "This version is no longer maintained. Please upgrade."
```

---

## ✅ Bonus: Test Before Publishing

Use:

```bash
npm pack --dry-run
```

To preview **exactly what files** will be published. Saves a lot of mistakes. ✅

---

## 🧠 Summary Table

| Error Message             | Cause                             | Fix                                             |
| ------------------------- | --------------------------------- | ----------------------------------------------- |
| `403 Forbidden`           | Wrong ownership or scope          | Use scoped name or log in correctly             |
| Version already exists    | Trying to re-publish same version | Bump version with `npm version`                 |
| Scoped package is private | Missing `--access public`         | Add `--access public` when publishing           |
| No README on npm          | Missing or empty `README.md`      | Add meaningful `README.md`                      |
| Extra files published     | No `.npmignore` or `files` field  | Use `.npmignore` or `"files"` in `package.json` |
| Not logged in             | Forgot to `npm login`             | Run `npm login` and retry                       |
| Can't unpublish           | Over 72 hrs old                   | Use `npm deprecate` instead                     |

---

> 🧠 **In short**:
> Most NPM publishing errors are due to **missing metadata**, **wrong versioning**, or **access settings**. Run `npm whoami`, bump versions smartly, and always test with `npm pack --dry-run`.
