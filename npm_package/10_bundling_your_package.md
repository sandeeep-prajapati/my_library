### 📦 How Do You Bundle Your NPM Package with Tools Like Webpack, Rollup, or ESBuild?

---

When building an NPM package, it's a good idea to **bundle your source code** into a clean, production-ready format for publishing.

Bundlers like **Webpack**, **Rollup**, and **ESBuild** help you:

* 🎯 Generate smaller, optimized output
* 🔀 Bundle multiple modules into one or more files
* ⚙️ Handle TypeScript, ES modules, and polyfills
* 🔄 Produce both **CommonJS (`cjs`)** and **ESM (`esm`)** outputs

---

## 🧰 Choosing a Bundler

| Bundler     | Best For                             | Pros                                     |
| ----------- | ------------------------------------ | ---------------------------------------- |
| **Rollup**  | Libraries & NPM packages             | Tree-shaking, minimal bundles, ESM-first |
| **ESBuild** | Fast development & simple bundling   | Extremely fast, zero-config builds       |
| **Webpack** | Complex apps & plugin-heavy packages | Feature-rich, but heavier for libraries  |

For NPM packages, **Rollup or ESBuild** is preferred due to their simplicity and output size.

---

## ✅ Let’s See an Example with **Rollup** (Recommended for NPM Libraries)

---

### 📁 1. Folder Structure

```
my-package/
├── src/
│   └── index.ts
├── dist/
├── rollup.config.js
├── package.json
├── tsconfig.json
└── README.md
```

---

### 📦 2. Install Dependencies

```bash
npm install --save-dev rollup typescript @rollup/plugin-node-resolve @rollup/plugin-commonjs @rollup/plugin-typescript
```

---

### ⚙️ 3. Create `rollup.config.js`

```js
// rollup.config.js
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';

export default {
  input: 'src/index.ts',
  output: [
    {
      file: 'dist/index.cjs.js',
      format: 'cjs',
      sourcemap: true,
    },
    {
      file: 'dist/index.esm.js',
      format: 'esm',
      sourcemap: true,
    },
  ],
  plugins: [resolve(), commonjs(), typescript()],
  external: [] // Add external dependencies here to avoid bundling them
};
```

---

### 📜 4. Update `package.json`

```json
{
  "name": "@sandeep/my-lib",
  "main": "dist/index.cjs.js",
  "module": "dist/index.esm.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "rollup -c"
  },
  "devDependencies": {
    "rollup": "^3.0.0",
    "typescript": "^5.0.0",
    "@rollup/plugin-typescript": "^11.0.0",
    "@rollup/plugin-commonjs": "^24.0.0",
    "@rollup/plugin-node-resolve": "^15.0.0"
  }
}
```

---

### 🏗️ 5. Build the Package

```bash
npm run build
```

You will get:

```
dist/
├── index.cjs.js
├── index.esm.js
├── index.d.ts
```

---

## ⚡ Alternative: ESBuild (Super Fast)

Install:

```bash
npm install --save-dev esbuild
```

Add script:

```json
"scripts": {
  "build": "esbuild src/index.ts --bundle --outdir=dist --format=esm --target=es2015"
}
```

✅ Great for rapid builds, but **doesn't generate `.d.ts` files** by default. For that, pair with `tsc`:

```bash
tsc --emitDeclarationOnly --outDir dist
```

---

## 🛠️ Notes for All Bundlers

* Always **exclude dependencies** from the bundle using `external` in Rollup or `--external:` in ESBuild.
* Export your functions via `index.ts` cleanly.
* Keep `src/` for development and `dist/` for final output.
* Don’t publish `src/`, only the `dist/` folder.

---

## 🧪 Sample `src/index.ts`

```ts
export function greet(name: string): string {
  return `Hello, ${name}`;
}
```

---

> 🧠 **In short:** Use **Rollup** or **ESBuild** to bundle your TypeScript NPM package into optimized JS (ESM + CJS), generate `.d.ts` files, and configure your `package.json` properly.

Want a ready-made boilerplate with Rollup + TS + Jest + Build script?
