Great question, Sandeep! 💡 Whether you're solo or in a team, using Git efficiently with Bitbucket is key to smooth Laravel development. Here's a curated list of **essential Git commands every Laravel developer should know**:

---

## 🧠 **Essential Git Commands for Laravel Devs Using Bitbucket**

### 🔹 1. **Initialize a Git repo (if starting from scratch)**
```bash
git init
```

---

### 🔹 2. **Clone an existing Bitbucket repository**
```bash
git clone git@bitbucket.org:username/repo-name.git
```

---

### 🔹 3. **Check repository status**
```bash
git status
```

---

### 🔹 4. **Add files to staging**
```bash
git add .
# or for specific files
git add filename.php
```

---

### 🔹 5. **Commit your changes**
```bash
git commit -m "Meaningful message"
```

---

### 🔹 6. **Push to Bitbucket**
```bash
git push origin main
# or for a feature branch
git push origin feature/your-branch
```

---

### 🔹 7. **Pull the latest changes**
```bash
git pull origin main
```

---

### 🔹 8. **Create a new branch for features/bugs**
```bash
git checkout -b feature/new-ui
```

---

### 🔹 9. **Switch between branches**
```bash
git checkout main
```

---

### 🔹 10. **View branches**
```bash
git branch
```

---

### 🔹 11. **Merge a feature branch into `main`**
```bash
git checkout main
git merge feature/new-ui
```

---

### 🔹 12. **Delete a local branch (after merge)**
```bash
git branch -d feature/new-ui
```

---

### 🔹 13. **Stash uncommitted changes temporarily**
```bash
git stash
```

Restore them later:
```bash
git stash pop
```

---

### 🔹 14. **Show commit history**
```bash
git log --oneline --graph --all
```

---

### 🔹 15. **View a specific file's history**
```bash
git log path/to/file.php
```

---

### 🔹 16. **Undo the last commit (but keep changes)**
```bash
git reset --soft HEAD~1
```

---

### 🔹 17. **Revert a committed change**
```bash
git revert <commit-hash>
```

---

### 🔹 18. **Tag a release version**
```bash
git tag -a v1.0 -m "First stable release"
git push origin v1.0
```

---

### 🔹 19. **Resolve merge conflicts**
When Git detects a conflict:
```bash
# Open conflicting file, fix manually
git add filename.php
git commit -m "Resolve merge conflict"
```

---

### 🔹 20. **Check remote URLs**
```bash
git remote -v
```

---

Want me to generate a printable cheat sheet or give the next Laravel–Bitbucket integration prompt? (e.g., **"How to set up Bitbucket Pipelines for Laravel CI/CD"**?)