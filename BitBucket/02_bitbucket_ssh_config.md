Here's how you can **configure SSH keys** to securely connect your **local Laravel development environment** with **Bitbucket**:

---

## 🔐 **How to Configure SSH Keys for Bitbucket (Laravel Setup)**

### ✅ 1. **Check for Existing SSH Keys**
Open your terminal and run:
```bash
ls -al ~/.ssh
```
> Look for files like `id_rsa` and `id_rsa.pub`.

If keys **already exist**, skip to step 3.

---

### ✅ 2. **Generate a New SSH Key (if needed)**
```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```
- When prompted, press `Enter` to accept default file path.
- You can optionally set a passphrase for extra security.

This creates:
- `~/.ssh/id_rsa` (private key)
- `~/.ssh/id_rsa.pub` (public key)

---

### ✅ 3. **Start SSH Agent and Add Your Key**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

---

### ✅ 4. **Copy the Public Key**
```bash
cat ~/.ssh/id_rsa.pub
```
- Copy the full output (starting with `ssh-rsa`...)

---

### ✅ 5. **Add the SSH Key to Bitbucket**
- Go to: [Bitbucket → Personal settings → SSH keys](https://bitbucket.org/account/settings/ssh-keys/)
- Click **“Add key”**
- Paste your public key
- Name it (e.g., “My Laravel Dev Machine”)
- Click **Add key**

---

### ✅ 6. **Test the SSH Connection**
```bash
ssh -T git@bitbucket.org
```
You should see:
```
authenticated via ssh key.
You can use git to connect to Bitbucket.
```

---

### ✅ 7. **Use SSH URLs for Cloning**
Instead of using HTTPS URLs like:
```
https://bitbucket.org/username/laravel-app.git
```
Use the **SSH version**:
```
git@bitbucket.org:username/laravel-app.git
```

---

### 🎉 You're Done!

Your Laravel development environment is now securely connected to Bitbucket via SSH — no more typing usernames or passwords when pushing or pulling.

Would you like the next step? I can guide you on **setting up `.gitignore` for Laravel** or **enabling Bitbucket Pipelines**.