
---

## **1. Installing Docker on Windows**
### **System Requirements**
- Windows 10/11 (64-bit) with WSL 2 (Windows Subsystem for Linux 2) enabled.
- Hyper-V and Containers features must be enabled.

### **Installation Steps**
1. **Enable WSL 2** (if not already enabled):
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```
   Restart your PC.

2. **Download Docker Desktop for Windows**:
   - Get the installer from [Docker's official site](https://www.docker.com/products/docker-desktop/).
   - Run the installer and follow the prompts.

3. **Launch Docker Desktop**:
   - After installation, Docker starts automatically.
   - Verify installation by running in PowerShell/CMD:
     ```bash
     docker --version
     docker run hello-world
     ```

### **Common Issues & Fixes**
- **"Docker Desktop requires WSL 2"**: Ensure WSL 2 is enabled and set as default.
- **Hyper-V not enabled**: Run in PowerShell as Admin:
  ```powershell
  Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
  ```
- **Slow performance**: Allocate more resources in Docker Desktop settings (Resources → WSL Integration).

---

## **2. Installing Docker on macOS**
### **System Requirements**
- macOS 10.15 (Catalina) or later (Intel or Apple Silicon).

### **Installation Steps**
1. **Download Docker Desktop for Mac**:
   - Get it from [Docker's official site](https://www.docker.com/products/docker-desktop/).
   - Open the `.dmg` file and drag Docker to Applications.

2. **Run Docker Desktop**:
   - Open from Applications.
   - Grant necessary permissions when prompted.

3. **Verify Installation**:
   ```bash
   docker --version
   docker run hello-world
   ```

### **Common Issues & Fixes**
- **"Cannot connect to Docker daemon"**: Restart Docker Desktop.
- **High CPU/Memory usage**: Adjust resources in Docker Desktop (Preferences → Resources).
- **Apple Silicon (M1/M2) issues**: Use `--platform linux/amd64` flag for x86 images:
  ```bash
  docker run --platform linux/amd64 hello-world
  ```

---

## **3. Installing Docker on Linux (Ubuntu/Debian)**
### **System Requirements**
- 64-bit Linux (Ubuntu/Debian/CentOS/etc.).
- `curl` and `systemd` should be installed.

### **Installation Steps**
1. **Uninstall old versions (if any)**:
   ```bash
   sudo apt remove docker docker-engine docker.io containerd runc
   ```

2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y ca-certificates curl gnupg
   ```

3. **Add Docker’s GPG key**:
   ```bash
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   ```

4. **Add Docker repository**:
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. **Install Docker Engine**:
   ```bash
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

6. **Start Docker and verify**:
   ```bash
   sudo systemctl enable --now docker
   sudo docker run hello-world
   ```

7. **Run Docker without `sudo` (optional)**:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker  # Re-login or restart terminal
   ```

### **Common Issues & Fixes**
- **"Permission denied"**: Ensure user is in the `docker` group (`sudo usermod -aG docker $USER`).
- **"Cannot connect to Docker daemon"**: Restart Docker:
  ```bash
  sudo systemctl restart docker
  ```
- **"No space left on device"**: Clean up unused containers/images:
  ```bash
  docker system prune -a
  ```

---

## **Initial Configuration (All Platforms)**
1. **Check Docker status**:
   ```bash
   docker info
   ```
2. **Test a container**:
   ```bash
   docker run -it ubuntu bash
   ```
3. **Configure Docker to start on boot**:
   - Linux: `sudo systemctl enable docker`
   - Windows/macOS: Enable in Docker Desktop settings.

---

## **Final Notes**
- **Windows/macOS**: Use Docker Desktop for a GUI experience.
- **Linux**: Prefer CLI (`docker`, `docker-compose`).
- **Troubleshooting**:
  - Check logs: `docker logs <container-id>`
  - Reset Docker (Desktop): Troubleshoot → Reset to factory defaults.
