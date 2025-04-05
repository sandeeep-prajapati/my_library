
---

## 🛡️ **How to Set Up Branch Protection Rules in Bitbucket for Laravel's `main` and `develop` Branches**

---

### 🚀 **Why Protect Branches?**

- Prevent direct pushes to `main` or `develop`
- Enforce code reviews
- Ensure Laravel app is tested before merging (via pipelines)
- Avoid broken code in production or staging

---

### ✅ 1. **Go to Your Bitbucket Repo Settings**

1. Navigate to your repo:  
   `https://bitbucket.org/<your-team>/<your-laravel-repo>/`
2. Click on ⚙️ **Repository settings**
3. Under **Workflow**, click **Branch permissions**

---

### ✅ 2. **Click "Add a branch permission"**

You’ll do this **twice** — once for `main`, once for `develop`.

---

### ✅ 3. **Protect the `main` Branch**

| Field                   | Value                                 |
|------------------------|---------------------------------------|
| **Branch name pattern**| `main`                                |
| **Write access**       | Only your DevOps/Lead or empty        |
| ✅ Require pull request | ✔️ Enabled                          |
| ✅ Minimum approvals    | 1–2 reviewers                        |
| ✅ No changes without approval | ✔️                            |
| ✅ Prevent rewriting history | ✔️ (optional, for safety)       |

Click **Save**.

---

### ✅ 4. **Protect the `develop` Branch**

Same steps, but a bit more flexible:

| Field                   | Value                                |
|------------------------|--------------------------------------|
| **Branch name pattern**| `develop`                            |
| **Write access**       | Dev team lead or reviewers only      |
| ✅ Require pull request | ✔️ Enabled                         |
| ✅ Minimum approvals    | 1                                    |
| ✅ Successful pipeline  | ✔️ (if using Bitbucket Pipelines)   |

Click **Save**.

---

### ✅ 5. (Optional) Enforce Naming Conventions for Feature Branches

Add a rule like:

- **Branch pattern**: `feature/*`
- Allow **everyone** to write
- No protection needed here

---

## 🔒 Example Workflow for Laravel

| Branch       | Purpose             | Rule                                    |
|--------------|---------------------|------------------------------------------|
| `main`       | Production-ready     | Fully protected, no direct push allowed  |
| `develop`    | QA/Staging-ready     | PR + 1 approval required                 |
| `feature/*`  | Development           | Open for all, merged via PR              |
| `hotfix/*`   | Urgent bugfixes      | Short-lived, PR to `main`                |

---

## ⚙️ Tips for Laravel Projects

- Add **Laravel CI** tests in Bitbucket Pipelines and enforce successful builds
- Use **Slack or Discord** webhooks for PR notifications
- Combine with `.editorconfig` and `php-cs-fixer` to enforce formatting

---
