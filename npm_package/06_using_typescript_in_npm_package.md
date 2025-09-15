### 📦 How Do You Create a TypeScript-Based NPM Package and Generate Declaration Files (`.d.ts`)?

---

Creating a **TypeScript-based NPM package** ensures your code is **type-safe**, **self-documented**, and offers **autocomplete and type hints** to users of your package.

The key step in this process is generating **`.d.ts` (declaration files)**, which describe the types of your package to consumers (even if they’re using JavaScript).

---

### ✅ Step-by-Step Guide

---

### 📁 1. **Project Structure**

```
my-typescript-package/
├── src/
│   └── index.ts
├── dist/               # Compiled output
├── package.json
├── tsconfig.json
├── README.md
├── .gitignore
└── LICENSE
```

---

### 🧱 2. **Initialize the Package**

```bash
npm init -y
```

This creates a `package.json`. You’ll modify it later.

---

### 🔧 3. **Install TypeScript as a Dev Dependency**

```bash
npm install typescript --save-dev
```

You may also want tools like:

```bash
npm install rimraf --save-dev   # for cleaning dist/
```

---

### 🧾 4. **Create `tsconfig.json`**

Generate the config file:

```bash
npx tsc --init
```

Then update it to support declaration files:

```jsonc
{
  "compilerOptions": {
    "target": "ES2015",
    "module": "ESNext",
    "moduleResolution": "Node",
    "declaration": true,             // ✅ Generates .d.ts files
    "outDir": "./dist",              // ✅ Compiled output goes here
    "rootDir": "./src",              // ✅ Source files
    "strict": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "test"]
}
```

---

### 🧑‍💻 5. **Write Your TypeScript Code**

```ts
// src/index.ts
export function greet(name: string): string {
  return `Hello, ${name}!`;
}
```

---

### 🔨 6. **Build the Package**

Run:

```bash
npx tsc
```

This compiles `.ts` files into `dist/`:

```
dist/
├── index.js
└── index.d.ts
```

Now your package includes both the JS code and its type definitions!

---

### 📦 7. **Update `package.json`**

Make sure the compiled files are referenced properly:

```json
{
  "name": "@sandeep/my-typescript-utils",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "rimraf dist && tsc"
  },
  "license": "MIT",
  "devDependencies": {
    "typescript": "^5.0.0",
    "rimraf": "^5.0.0"
  }
}
```

---

### 🚀 8. **Test Locally**

To test your package before publishing:

```bash
npm link     # Inside your package directory
# Then in another local project:
npm link @sandeep/my-typescript-utils
```

Then:

```ts
import { greet } from '@sandeep/my-typescript-utils';
console.log(greet("Sandeep"));  // Hello, Sandeep!
```

---

### 🧪 Optional: Add a `test/` Folder

```ts
// test/index.test.ts
import { greet } from '../src';

console.assert(greet("Sandeep") === "Hello, Sandeep!");
```

---

### 📝 9. **.gitignore and .npmignore**

`.gitignore`:

```
node_modules
dist
```

Use `.npmignore` to exclude unnecessary files (or use `files` field in `package.json`).

---

### ✅ Summary

| Step                 | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `tsconfig.json`      | Enables declaration file generation with `"declaration": true` |
| `outDir` & `rootDir` | Define `src/` and `dist/` locations                            |
| `main` & `types`     | Points consumers to the JS and type files                      |
| `tsc`                | Compiles TypeScript to JS + `.d.ts` files                      |
| `npm publish`        | Publishes everything in `dist/` to NPM                         |

---

> 🧠 In short: Use TypeScript with the correct `tsconfig.json` settings to generate `.d.ts` files, and ensure `main` + `types` are correctly set in `package.json`.
