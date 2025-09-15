### 📁 What is the Recommended Folder Structure for a Reusable NPM Package?

---

Creating a clean and maintainable folder structure is essential when building a reusable NPM package. It helps you:

* Keep your codebase organized
* Separate concerns like source code, tests, builds, and documentation
* Make it easier for contributors (and your future self) to understand and use the package

---

### ✅ **Standard Folder Structure**

Here’s a commonly recommended folder structure for a modern, production-ready NPM package (with TypeScript):

```
my-awesome-package/
├── src/                  # All source code (modules, functions, classes)
│   └── index.ts
│
├── dist/                 # Compiled output (auto-generated, ignored in Git)
│   └── index.js
│
├── test/                 # Unit or integration tests
│   └── index.test.ts
│
├── node_modules/         # Installed dependencies (auto-generated)
│
├── .gitignore            # Ignore node_modules, dist, etc.
├── .npmignore            # Ignore files/folders during publish (if needed)
├── package.json          # NPM metadata and config
├── tsconfig.json         # TypeScript configuration
├── jest.config.js        # Jest testing config (or vitest.config.ts etc.)
├── README.md             # Documentation
├── LICENSE               # Open-source license
```

---

### 🧱 **Explanation of Each Folder/File**

| Item           | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| `src/`         | Your actual source code lives here (e.g., utility functions, classes, logic).                        |
| `dist/`        | The transpiled JavaScript files (using Babel, TypeScript, etc.). This is what gets published to NPM. |
| `test/`        | Unit tests using Jest, Mocha, Vitest, etc. Should mirror the structure of `src/`.                    |
| `.gitignore`   | Prevents clutter in your Git repo (ignore `node_modules`, `dist`, `.env`, etc.).                     |
| `.npmignore`   | Controls what files are excluded when publishing to NPM (if not using `"files"` in `package.json`).  |
| `package.json` | Contains package metadata, dependencies, and scripts.                                                |
| `README.md`    | Documentation about the package: how to install, use, and contribute.                                |
| `LICENSE`      | Specifies the usage rights (MIT, Apache-2.0, etc.).                                                  |

---

### 🧪 **Optional Enhancements**

* **`examples/`** → Code samples showing how to use your package.
* **`docs/`** → Auto-generated API docs via tools like `typedoc`.
* **`scripts/`** → Custom Node scripts (e.g., CLI tools).
* **`.github/`** → GitHub workflows, issue templates, PR templates.

---

### 📦 Publishing Tip

When publishing your package, make sure only production-ready files (like the `dist/` folder) are included. You can control this using:

1. `.npmignore` file
2. or the `files` field in `package.json`:

```json
"files": [
  "dist/",
  "README.md"
]
```

---

