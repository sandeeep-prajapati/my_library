### 🧳 What Are Scoped Packages in NPM, and How Do You Create One?

---

## 📦 What Is a **Scoped Package**?

A **scoped package** in NPM is a package that belongs to a **namespace**, typically representing a user or organization.

> 📛 Format: `@scope/package-name`
> ✅ Example: `@sandeep/my-utils`

---

### 🔍 Why Use Scoped Packages?

| Benefit                         | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| ✅ **Avoid name conflicts**      | You can use names that are already taken (e.g., `utils`)      |
| 👥 **Organization control**     | Group all your packages under one scope                       |
| 🔒 **Publish private packages** | Scoped packages support private publishing                    |
| 📁 **Better structure**         | Especially helpful when maintaining multiple related packages |

---

## ⚙️ How to Create a Scoped Package

---

### ✅ 1. Name Your Package with a Scope

In your `package.json`, set the name like this:

```json
{
  "name": "@sandeep/my-utils",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "license": "MIT"
}
```

> Replace `@sandeep` with your **npm username** or **organization name**.

---

### ✅ 2. Initialize Git (Recommended)

```bash
git init
git remote add origin https://github.com/sandeeep-prajapati/my-utils.git
```

---

### ✅ 3. Log In to NPM

```bash
npm login
```

---

### ✅ 4. Publish the Package

#### 🔓 For Public Scoped Packages:

```bash
npm publish --access public
```

> 📌 **IMPORTANT**:
> Scoped packages are **private by default**, so you must specify `--access public` to publish them **publicly**.

#### 🔒 For Private Packages:

```bash
npm publish --access restricted
```

> Requires a **paid npm account or organization** for private package hosting.

---

### 🧪 5. Install & Use the Scoped Package

```bash
npm install @sandeep/my-utils
```

Then in your code:

```ts
import { greet } from '@sandeep/my-utils';

console.log(greet('Sandeep'));
```

---

## 📁 Optional: Scoped Directory Structure

If you're creating **multiple packages** under a scope (monorepo or Lerna), your folder structure might look like:

```
packages/
├── my-utils/
│   └── package.json (name: @sandeep/my-utils)
├── string-tools/
│   └── package.json (name: @sandeep/string-tools)
```

---

## 🧠 Summary

| Feature                    | Scoped Package           |
| -------------------------- | ------------------------ |
| Naming                     | `@username/package-name` |
| Private by default         | ✅ Yes                    |
| Requires `--access public` | ✅ Yes                    |
| Useful for orgs            | ✅ Yes                    |
| Avoid name conflicts       | ✅ Yes                    |

---

> 🧠 **In short**:
> Scoped packages let you publish under a personal or org namespace like `@sandeep/utils`. They’re private by default and require `--access public` for public sharing.
