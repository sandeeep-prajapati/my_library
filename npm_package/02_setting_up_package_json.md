
---

#### 🧱 **Step 1: Initialize a New NPM Package**

To start building a new NPM package, you need to create a `package.json` file, which acts as the **manifest** for your package.

Run this command in your terminal:

```bash
npm init
```

This command walks you through a step-by-step prompt asking for:

* **Package name** (must be unique on NPM if publishing publicly)
* **Version** (`1.0.0` by default)
* **Description** (what your package does)
* **Entry point** (`index.js` by default)
* **Test command** (e.g., `jest`)
* **Git repository** (if open source or version-controlled)
* **Keywords** (to improve discoverability)
* **Author** (your name or org)
* **License** (e.g., MIT, ISC)

You can skip the interactive wizard and generate a default config using:

```bash
npm init -y
```

This creates a basic `package.json` file like:

```json
{
  "name": "my-awesome-package",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"No test specified\" && exit 0"
  },
  "keywords": [],
  "author": "Sandeep Prajapati",
  "license": "ISC"
}
```

---

#### 🛠️ **Step 2: Configure `package.json` Properly**

Here are some essential and optional fields you may want to configure:

| Field             | Description                                                               |
| ----------------- | ------------------------------------------------------------------------- |
| `name`            | Unique package name (avoid using capital letters or spaces).              |
| `version`         | Follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.  |
| `main`            | Entry point file that consumers import (e.g., `"main": "dist/index.js"`). |
| `types`           | Path to your TypeScript declaration file (`.d.ts`), if using TypeScript.  |
| `scripts`         | CLI commands (e.g., `build`, `test`, `lint`).                             |
| `repository`      | GitHub/GitLab repo link for tracking and contributions.                   |
| `keywords`        | Array of strings for discoverability on npmjs.com.                        |
| `license`         | Licensing info (`MIT`, `Apache-2.0`, etc.).                               |
| `dependencies`    | Runtime dependencies.                                                     |
| `devDependencies` | Development-only dependencies (e.g., TypeScript, Jest).                   |
| `engines`         | Node.js version compatibility.                                            |

---

#### ✅ **Example `package.json` for a TypeScript NPM Package**

```json
{
  "name": "@sandeep/my-utils",
  "version": "1.0.0",
  "description": "Utility functions for data formatting",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/sandeeep-prajapati/my-utils.git"
  },
  "keywords": ["utils", "npm", "typescript"],
  "author": "Sandeep Prajapati",
  "license": "MIT",
  "devDependencies": {
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  },
  "engines": {
    "node": ">=14.0.0"
  }
}
```

---

#### 🧪 **Verify Everything Works**

You can run:

```bash
npm run build
npm run test
```

To ensure your setup works. Then you're ready to move on to development and publishing!

---
