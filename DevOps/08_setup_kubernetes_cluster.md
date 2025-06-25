# **Setting Up a Kubernetes Cluster: Minikube vs K3s**

Both **Minikube** (for local development) and **K3s** (for lightweight production) are popular ways to run Kubernetes. Below is a step-by-step guide for each.

---

## **Option 1: Minikube (Local Development Cluster)**
### **1. Install Prerequisites**
- **Docker** (or another driver like VirtualBox/Hyper-V)
- **kubectl** (Kubernetes CLI)
- **Minikube**

#### **Installation Commands:**
```bash
# Install kubectl (Linux/macOS)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Minikube (Linux/macOS)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
*(Windows: Use [Chocolatey](https://chocolatey.org/) or download binaries manually.)*

### **2. Start Minikube Cluster**
```bash
# Start with Docker driver (recommended)
minikube start --driver=docker

# Verify
minikube status
kubectl get nodes
```
Expected output:
```
NAME       STATUS   ROLES           AGE   VERSION
minikube   Ready    control-plane   10s   v1.28.0
```

### **3. Deploy a Test App**
```bash
# Run an Nginx deployment
kubectl create deployment nginx --image=nginx

# Expose it as a service
kubectl expose deployment nginx --port=80 --type=NodePort

# Access the app
minikube service nginx
```
- This opens Nginx in your default browser.

### **4. Stop/Minikube Cleanup**
```bash
minikube stop  # Pause the cluster
minikube delete  # Destroy it
```

---

## **Option 2: K3s (Lightweight Production Cluster)**
### **1. Install K3s (Single-Node Cluster)**
```bash
# Install K3s (automatically starts a cluster)
curl -sfL https://get.k3s.io | sh -

# Verify
sudo k3s kubectl get nodes
```
Expected output:
```
NAME     STATUS   ROLES                  AGE   VERSION
k3s-01   Ready    control-plane,master   30s   v1.28.0+k3s
```

### **2. Access the Cluster**
K3s stores its config at `/etc/rancher/k3s/k3s.yaml`. To use `kubectl`:
```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get pods -A
```

### **3. Deploy a Test App**
```bash
# Deploy Nginx
kubectl create deployment nginx --image=nginx

# Expose it
kubectl expose deployment nginx --port=80 --type=NodePort

# Get the service URL
kubectl get svc nginx
```
Access via:
```bash
curl http://<NODE_IP>:<NodePort>
```

### **4. Multi-Node Setup (Optional)**
On **worker nodes**, run:
```bash
curl -sfL https://get.k3s.io | K3S_URL=https://<MASTER_IP>:6443 K3S_TOKEN=<NODE_TOKEN> sh -
```
- `<NODE_TOKEN>` is found at `/var/lib/rancher/k3s/server/node-token` on the master.

### **5. Uninstall K3s**
```bash
/usr/local/bin/k3s-uninstall.sh
```

---

## **Key Differences: Minikube vs K3s**
| Feature               | **Minikube**                          | **K3s**                              |
|-----------------------|---------------------------------------|--------------------------------------|
| **Purpose**          | Local development                     | Lightweight production/edge          |
| **Resource Usage**   | Higher (runs a VM)                    | Lower (runs directly on host)        |
| **Setup Complexity** | Simple (single command)               | Simple, but multi-node needs config  |
| **Networking**       | Limited (local-only)                  | Supports real-world networking       |
| **Best For**         | Learning, local testing               | Raspberry Pi, IoT, small production  |

---

## **Final Recommendations**
- **Use Minikube if**:  
  You need a quick local Kubernetes environment for development/testing.  

- **Use K3s if**:  
  You want a lightweight, production-ready cluster (e.g., for homelabs, edge computing).  

Both tools simplify Kubernetes, but **K3s is closer to real-world deployments**, while **Minikube is ideal for beginners**.  

🚀 **Now you’re ready to experiment with Kubernetes!**