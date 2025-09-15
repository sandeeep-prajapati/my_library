### 🧩 How Can You Write Modular and Reusable Code Suitable for Publishing as a Package?

---

Creating an NPM package means your code will be **used by others**, possibly in **diverse environments** and **different use cases**. To make your code modular and reusable:

---

### 🧱 1. **Follow the Single Responsibility Principle (SRP)**

> Each module/function should do **one thing** and do it well.

✅ Break large functionalities into smaller, focused utilities or classes.

**Example:**

```ts
// src/math/add.ts
export function add(a: number, b: number): number {
  return a + b;
}
```

---

### 📦 2. **Use Named Exports for Maximum Flexibility**

Use **named exports** so users can import only what they need.

```ts
// src/index.ts
export * from './math/add';
export * from './math/subtract';
```

Usage:

```ts
import { add } from '@sandeep/my-utils';
```

Avoid default exports in libraries unless exporting a single entity.

---

### 💼 3. **Keep Code Framework-Agnostic**

Do not tie your code to frameworks (e.g., React, Express) unless your package is framework-specific. This maximizes reusability.

---

### 🧪 4. **Write Pure Functions Wherever Possible**

> Pure functions return the same output for the same input and have no side effects.

They are:

* Easier to test
* Easier to reuse
* More predictable

---

### 🔁 5. **Parameterize Configs Instead of Hardcoding**

Allow users to pass options or settings, rather than hardcoding values.

```ts
export function greet(name: string, options = { capitalize: false }): string {
  return options.capitalize ? `Hello, ${name.toUpperCase()}` : `Hello, ${name}`;
}
```

---

### 📜 6. **Use TypeScript or JSDoc for Type Safety**

Ensure your code is well-typed for better developer experience and IDE autocompletion.

```ts
export function isEven(n: number): boolean {
  return n % 2 === 0;
}
```

---

### 🔧 7. **Make It Tree-Shakable**

Avoid side effects and make use of **ES modules** so bundlers can tree-shake unused code.

> ✅ No side-effects = smaller builds for consumers

---

### 🧩 8. **Design for Composition**

Make utilities composable. Instead of tightly coupling functions, let them work together flexibly.

```ts
const input = [1, 2, 3];
const doubledEvens = input.filter(isEven).map(double);
```

---

### 📁 9. **Structure Code by Feature/Domain**

Group related functions into folders, e.g., `math/`, `string/`, `date/`, etc. This improves scalability.

```
src/
├── math/
│   ├── add.ts
│   └── subtract.ts
├── string/
│   └── capitalize.ts
```

---

### 🧹 10. **Avoid Global State or Side Effects**

Avoid things like modifying global variables, logging directly, or reading/writing files unless necessary.

Instead of:

```ts
console.log("Hello"); // ❌
```

Do:

```ts
export function getGreeting(name: string): string {
  return `Hello, ${name}`;
}
```

---

### 📚 Example: A Simple Utility Package

```ts
// src/string/capitalize.ts
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// src/index.ts
export * from './string/capitalize';
```

Consumers can now:

```ts
import { capitalize } from '@sandeep/my-utils';
```

---

> ✅ **In Summary**: Write **small**, **well-named**, **pure**, **typed**, and **framework-independent** functions, organize them logically, and export them cleanly for easy reuse.

