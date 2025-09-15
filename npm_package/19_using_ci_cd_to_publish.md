### 🤖 How Do You Set Up **GitHub Actions** or **GitLab CI** to Automate Publishing of Your NPM Package?

---

Automating your NPM package publishing with **CI/CD** (like GitHub Actions or GitLab CI) ensures that your package is **consistently built, tested, and published** — without manual steps. Great for open source, internal tools, and monorepos.

---

## 🚀 Option 1: Using **GitHub Actions**

### 📁 Project Structure (Basic)

```
.my-npm-package/
├── .github/
│   └── workflows/
│       └── publish.yml
├── dist/
├── src/
├── package.json
└── README.md
```

---

### 🔐 Step 1: Add NPM Token to GitHub Secrets

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Add a new secret:

   * Name: `NPM_TOKEN`
   * Value: *(Your token from [https://www.npmjs.com/settings/YOUR\_USERNAME/tokens](https://www.npmjs.com/settings/YOUR_USERNAME/tokens))*

---

### 🛠️ Step 2: Create `.github/workflows/publish.yml`

```yaml
name: 🚀 Publish to NPM

on:
  push:
    tags:
      - 'v*' # Triggers only on version tags like v1.0.0

jobs:
  publish:
    runs-on: ubuntu-latest

    steps:
      - name: ⬇️ Checkout code
        uses: actions/checkout@v4

      - name: 🔧 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org/'

      - name: 📦 Install dependencies
        run: npm ci

      - name: 🧱 Build the package
        run: npm run build

      - name: 🚀 Publish to NPM
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

> ✅ This workflow:
>
> * Runs only when you push a tag like `v1.0.0`
> * Builds and publishes your package to NPM using the stored token

---

### 🏷️ Step 3: Release a New Version

```bash
npm version patch   # or minor / major
git push --follow-tags
```

CI/CD will handle the rest 🚀

---

## 🦊 Option 2: Using **GitLab CI**

### 🔐 Step 1: Add NPM Token as CI/CD Variable

1. Go to your project → **Settings** → **CI/CD** → **Variables**
2. Add:

   * Key: `NPM_TOKEN`
   * Value: *(your NPM access token)*

---

### 🛠️ Step 2: Add `.gitlab-ci.yml`

```yaml
stages:
  - build
  - publish

build:
  stage: build
  image: node:18
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/

publish:
  stage: publish
  image: node:18
  only:
    - tags
  script:
    - echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" > ~/.npmrc
    - npm publish --access public
```

> 📦 Only runs on tagged releases.

---

## 🛡️ Secure Publishing Best Practices

| Tip                               | Why                          |
| --------------------------------- | ---------------------------- |
| Use `NODE_AUTH_TOKEN` or `.npmrc` | Never commit tokens directly |
| Use semantic versioning + tags    | Helps CI detect releases     |
| Run tests before publish          | Prevent broken versions      |
| Use `.npmrc` with token in CI     | Allows headless publishing   |

---

## ✅ Summary

| Platform | File                            | Trigger       | Secure Token        |
| -------- | ------------------------------- | ------------- | ------------------- |
| GitHub   | `.github/workflows/publish.yml` | `push → tags` | `secrets.NPM_TOKEN` |
| GitLab   | `.gitlab-ci.yml`                | `tags only`   | `$NPM_TOKEN`        |

---

> 🧠 **In short**:
> Automated publishing ensures consistency, saves time, and avoids mistakes. Use GitHub Actions or GitLab CI with `NPM_TOKEN` and version tags to securely build and publish every release.
