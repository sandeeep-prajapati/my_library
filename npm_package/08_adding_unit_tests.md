### 🧪 How Do You Write Unit Tests for Your NPM Package Using Jest or Mocha?

---

Writing unit tests ensures your NPM package is **reliable**, **maintainable**, and **safe to update**. Popular frameworks for JavaScript/TypeScript testing include:

* ✅ **Jest** – All-in-one, zero-config, most popular.
* ✅ **Mocha** – Modular, flexible, often used with Chai for assertions.

---

## ✅ Example Using **Jest** (Recommended for Most Projects)

---

### 📦 1. **Install Jest**

If you're using **JavaScript**:

```bash
npm install --save-dev jest
```

If you're using **TypeScript**:

```bash
npm install --save-dev jest ts-jest @types/jest
```

---

### 🧾 2. **Configure Jest**

Create a `jest.config.js` (or use `npx ts-jest config:init` for TS):

```js
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/test'],
};
```

---

### 🧪 3. **Write a Test File**

#### Your function (e.g., `src/greet.ts`):

```ts
export function greet(name: string): string {
  return `Hello, ${name}!`;
}
```

#### Your test file (e.g., `test/greet.test.ts`):

```ts
import { greet } from '../src/greet';

describe('greet()', () => {
  it('should return a greeting message', () => {
    expect(greet('Sandeep')).toBe('Hello, Sandeep!');
  });
});
```

---

### ▶️ 4. **Run Tests**

```bash
npm test
```

Or add a test script to `package.json`:

```json
"scripts": {
  "test": "jest"
}
```

---

## 🧪 Example Using **Mocha + Chai**

---

### 📦 1. **Install Mocha & Chai**

```bash
npm install --save-dev mocha chai
```

For TypeScript:

```bash
npm install --save-dev ts-node @types/mocha @types/chai
```

---

### 🧾 2. **Write a Test File**

#### Your function (e.g., `src/math.ts`):

```ts
export function add(a: number, b: number): number {
  return a + b;
}
```

#### Your test (e.g., `test/math.test.ts`):

```ts
import { expect } from 'chai';
import { add } from '../src/math';

describe('add()', () => {
  it('should return the sum of two numbers', () => {
    expect(add(2, 3)).to.equal(5);
  });
});
```

---

### ▶️ 3. **Run Tests with Mocha**

Add a test script to `package.json`:

```json
"scripts": {
  "test": "mocha -r ts-node/register test/**/*.test.ts"
}
```

Then:

```bash
npm test
```

---

## 🧹 Recommended Folder Structure

```
my-package/
├── src/
│   └── greet.ts
├── test/
│   └── greet.test.ts
├── package.json
├── tsconfig.json
├── jest.config.js
```

---

## ✅ Summary Table

| Tool  | Use For        | Config Required | Supports TypeScript |
| ----- | -------------- | --------------- | ------------------- |
| Jest  | All-in-one     | Minimal         | Yes (via `ts-jest`) |
| Mocha | Flexible setup | More config     | Yes (`ts-node`)     |

---

> 💡 **In short:**
> Use **Jest** for a fast, simple setup, or **Mocha + Chai** if you need more customization. Always keep tests in a `test/` directory and run them automatically with `npm test`.

