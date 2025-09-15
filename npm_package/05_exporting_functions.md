### 📤 How Do You Export a Function or Class from Your NPM Package So It Can Be Consumed by Other Projects?

---

To make your package usable, you must **explicitly export** functions, classes, or constants so that others can **import** them in their own projects.

---

### ✅ 1. **Basic Function Export**

#### File: `src/greet.ts`

```ts
// Named export
export function greet(name: string): string {
  return `Hello, ${name}`;
}
```

#### Usage:

```ts
import { greet } from 'your-package-name';
```

---

### ✅ 2. **Exporting a Class**

#### File: `src/Logger.ts`

```ts
export class Logger {
  log(message: string): void {
    console.log(`[LOG]: ${message}`);
  }
}
```

#### Usage:

```ts
import { Logger } from 'your-package-name';

const logger = new Logger();
logger.log('Package loaded');
```

---

### ✅ 3. **Exporting Multiple Things from One File**

#### File: `src/math.ts`

```ts
export function add(a: number, b: number): number {
  return a + b;
}

export function subtract(a: number, b: number): number {
  return a - b;
}
```

#### Usage:

```ts
import { add, subtract } from 'your-package-name';
```

---

### ✅ 4. **Re-exporting from an Index File**

Use `src/index.ts` (or `index.js`) to **collect and expose all public APIs**:

#### File: `src/index.ts`

```ts
export * from './greet';
export * from './Logger';
export * from './math';
```

This becomes the **main entry point** defined in `package.json`:

```json
{
  "main": "dist/index.js",
  "types": "dist/index.d.ts"
}
```

So now, users can just:

```ts
import { greet, Logger } from 'your-package-name';
```

---

### 🚫 5. **Avoid Default Exports in Libraries (Optional Advice)**

```ts
// Bad for libraries (can cause interop issues)
export default function greet() {}
```

Prefer **named exports**, as they:

* Improve tree-shaking (dead code elimination)
* Avoid confusion with mixing `require()` and `import`
* Provide better IntelliSense and auto-imports in IDEs

---

### 🛠 Build Step (if using TypeScript)

If you're using TypeScript, compile the files using:

```bash
tsc
```

This will output the `dist/` folder with `.js` and `.d.ts` files, which are the actual exports consumed by other projects.

---

### 📦 Example Usage in Consumer Project

```bash
npm install your-package-name
```

```ts
import { greet } from 'your-package-name';

console.log(greet('Sandeep')); // Hello, Sandeep
```

---

> ✅ In short: **Use named exports** in your source files and re-export them from a central `index.ts`. Ensure your `package.json` points to the built file via `main` and `types`.

