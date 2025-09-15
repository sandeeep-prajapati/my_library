### 🔁 How Do You **Update** Your Published NPM Package or **Unpublish** It (If Necessary)?

---

After publishing your NPM package, you might need to:

* 🚀 **Update** it (e.g., bug fix, new feature)
* ❌ **Unpublish** it (e.g., critical mistake, private data leak)

Here’s how to handle both safely and correctly 👇

---

## 🔁 PART 1: Updating Your NPM Package

---

### ✅ 1. **Make Your Changes**

Update your code, tests, docs, etc.

---

### ✅ 2. **Bump the Version (Required)**

> NPM **does not allow** publishing the same version twice.

Use **Semantic Versioning (SemVer)**:

```bash
npm version patch   # 1.0.0 → 1.0.1 (bug fix)
npm version minor   # 1.0.1 → 1.1.0 (new features)
npm version major   # 1.1.0 → 2.0.0 (breaking changes)
```

It will:

* Update `package.json`
* Create a Git tag (optional)

---

### ✅ 3. **Rebuild the Package (if needed)**

```bash
npm run build
```

Ensure your build artifacts (`dist/`, `.d.ts`, etc.) are up-to-date.

---

### ✅ 4. **Publish the New Version**

```bash
npm publish
```

Or for scoped packages:

```bash
npm publish --access public
```

---

### ✅ 5. **Done!**

Your package is now updated. 🎉
New users will get the latest version, while old users are still locked to their version unless they upgrade manually.

---

## ❌ PART 2: Unpublishing a Package

---

### ⚠️ Important NPM Rules

| Rule              | Details                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| 🔒 After 72 hours | You **cannot unpublish entire package**                                         |
| 🧱 Scoped version | You **can unpublish a specific version** anytime                                |
| 🛡️ Public safety | You cannot unpublish packages that are widely depended on (>300 downloads/week) |

---

### ✅ To Unpublish a Specific Version

```bash
npm unpublish @your-name/package-name@1.0.1
```

This removes **just that version**.

---

### 🚫 To Unpublish the Whole Package (within 72 hrs)

```bash
npm unpublish @your-name/package-name --force
```

⚠️ Use `--force` very carefully — this is irreversible.

---

### 📝 Tips Before Unpublishing

* Consider deprecating the package instead:

  ```bash
  npm deprecate @your-name/package-name@"<2.0.0" "This version is buggy. Please upgrade to 2.0.0."
  ```

* Fix the issue and publish a new patch version (preferred over unpublishing).

---

## 🔐 Common Use Cases

| Situation                               | Action                                                      |
| --------------------------------------- | ----------------------------------------------------------- |
| You published with sensitive data       | Unpublish ASAP (within 72 hours)                            |
| You published a broken version          | Deprecate + republish                                       |
| You renamed your package                | Unpublish old one if within 72 hours                        |
| You want to take down a private package | Consider switching to a private registry or GitHub Packages |

---

## ✅ Summary

| Task                        | Command                                          |
| --------------------------- | ------------------------------------------------ |
| Bump version                | `npm version patch/minor/major`                  |
| Publish update              | `npm publish`                                    |
| Unpublish version           | `npm unpublish @scope/package@version`           |
| Unpublish all (72 hrs only) | `npm unpublish --force`                          |
| Deprecate bad version       | `npm deprecate @scope/package@version "message"` |

---

> 🧠 **In short**:
> Use `npm version` to update and `npm publish` to release. Unpublish only within 72 hours, and consider `npm deprecate` instead for better safety and developer trust.

