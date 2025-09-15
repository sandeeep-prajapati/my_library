### 🧪 How Can You Test Your NPM Package Locally Using `npm link` or Local Paths?

---

Before publishing your NPM package to the registry, it's important to **test it locally** in a real project environment. There are two common ways to do this:

---

## 🔗 1. Testing Using `npm link` (Symlink Method)

`npm link` creates a **global symlink** to your local package that you can use in other projects.

### 📦 In your package folder:

```bash
npm link
```

This creates a global symlink from your package (e.g., `@sandeep/my-utils`) to your global NPM environment.

### 📁 In your test project folder:

```bash
npm link @sandeep/my-utils
```

This links your test project to the local package using the symlink created above.

### ✅ Now you can use the package:

```ts
import { greet } from '@sandeep/my-utils';
console.log(greet("Sandeep"));
```

---

### 🔁 To Update:

Every time you make changes to the package, **rebuild** it:

```bash
npm run build
```

No need to re-link; changes reflect automatically.

---

### ❌ To Unlink:

In your test project:

```bash
npm unlink @sandeep/my-utils
```

Globally (optional):

```bash
npm unlink --global @sandeep/my-utils
```

---

## 📁 2. Testing Using Local Path in `package.json`

You can install your package from a **local file path** (great for short-term testing without using symlinks).

### In your test project:

```bash
npm install ../my-utils
```

Or in `package.json`:

```json
{
  "dependencies": {
    "@sandeep/my-utils": "file:../my-utils"
  }
}
```

Then run:

```bash
npm install
```

### 🔄 Update Note:

* You must rebuild the package (`npm run build`)
* You must **reinstall** it using `npm install` to get updates (unlike `npm link`, which is live)

---

## ✅ Which One to Use?

| Method       | Pros                              | Cons                                                 |
| ------------ | --------------------------------- | ---------------------------------------------------- |
| `npm link`   | Instant updates, symlinked live   | Can cause issues with hoisting or TypeScript configs |
| `file:` path | Simple, no global symlinks needed | Requires reinstalling on change                      |

---

### 📌 Tip for TypeScript

If you're using TypeScript, make sure your package is **built** (`dist/` exists), and `package.json` has:

```json
"main": "dist/index.js",
"types": "dist/index.d.ts"
```

Then your consumer project will get full type support.

---

### ✅ Example Workflow

```bash
# In your package
npm run build
npm link

# In your test app
npm link @sandeep/my-utils
npm run dev  # or ts-node, vite, etc.
```

---

> 🧠 **In short**:
> Use `npm link` for **active development and live testing**, and use `file:` paths for **quick integration without symlinks**. Always remember to rebuild your package before testing it.
