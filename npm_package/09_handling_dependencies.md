### 📦 What is the Difference Between `dependencies`, `devDependencies`, and `peerDependencies` in NPM?

---

When creating an NPM package, understanding the difference between `dependencies`, `devDependencies`, and `peerDependencies` is crucial for:

* Keeping your package lean
* Avoiding version conflicts
* Ensuring the right tools are installed in the right context

Let’s break it down 👇

---

## 1. ✅ `dependencies`

These are the **packages your code needs to run** in **production**.

> 📦 They are installed automatically when someone installs your package.

### 📁 Example:

```json
"dependencies": {
  "lodash": "^4.17.21"
}
```

### ✅ Use when:

* You `import` or `require` this dependency in your **runtime code** (`src/`).
* The dependency is essential for your package to work.

### 🧠 Example use:

```ts
import _ from 'lodash';

export function isEmpty(value: any): boolean {
  return _.isEmpty(value);
}
```

---

## 2. 🛠️ `devDependencies`

These are packages used **only during development** or **build time**, such as:

* Testing tools (e.g., Jest, Mocha)
* Transpilers (e.g., TypeScript, Babel)
* Linters (e.g., ESLint)
* Bundlers (e.g., Rollup, Webpack)

> 🚫 They are **not installed** when someone installs your package from the registry.

### 📁 Example:

```json
"devDependencies": {
  "typescript": "^5.2.0",
  "jest": "^29.6.0"
}
```

### ✅ Use when:

* You need the package **only during development**, testing, or building.

---

## 3. 🤝 `peerDependencies`

These specify which versions of a package your package is **compatible with**, but don’t install them.

> ⚠️ They tell the **user** of your package:
> “You must install this dependency yourself.”

### 📁 Example:

```json
"peerDependencies": {
  "react": "^18.0.0"
}
```

### ✅ Use when:

* Your package needs to **plug into** another package (e.g., `React`, `Vue`, `jQuery`) but should **not install its own copy**.
* You’re building libraries, plugins, or UI components that **assume** the host project will already have that dependency installed.

### 🧠 Example:

You build a React component library:

```tsx
import React from 'react';

export const MyButton = () => <button>Click me</button>;
```

Instead of including `react` in `dependencies`, you add it as a `peerDependency`, so the host app uses **its own version** of React (avoids duplication and conflicts).

---

## 🧪 Summary Table

| Type               | Installed by Users?           | Used in Prod? | Typical Examples                  |
| ------------------ | ----------------------------- | ------------- | --------------------------------- |
| `dependencies`     | ✅ Yes                         | ✅ Yes         | `lodash`, `axios`, `chalk`        |
| `devDependencies`  | 🚫 No                         | ❌ No          | `jest`, `typescript`, `eslint`    |
| `peerDependencies` | 🚫 No (must install manually) | ⚠️ Depends    | `react`, `vue`, `webpack` plugins |

---

## 📌 Pro Tips

* ✅ Always **keep `devDependencies` out of the final build** (`dist/`).
* ⚠️ **Don't list both `dependencies` and `peerDependencies`** for the same package — it defeats the purpose of a peer.
* 🔍 Use `npm ls` and `npm outdated` to manage them cleanly.

---

> 🧠 **In short:**

* Use `dependencies` for packages needed at runtime.
* Use `devDependencies` for packages used only during development.
* Use `peerDependencies` when your package works alongside another, but shouldn't include its own copy.

