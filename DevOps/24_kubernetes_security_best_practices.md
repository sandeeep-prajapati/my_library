Securing a **Kubernetes cluster** is crucial for protecting workloads, infrastructure, and sensitive data. Here's how it can be done using **RBAC**, **Network Policies**, and **Pod Security**:

---

## ✅ 1. **Role-Based Access Control (RBAC)**

RBAC in Kubernetes is used to **control who can perform what actions** on which resources.

### 🔐 Key Concepts:

* **Roles/ClusterRoles**: Define sets of permissions.

  * `Role` is **namespace-scoped**
  * `ClusterRole` is **cluster-wide**
* **RoleBindings/ClusterRoleBindings**: Assign roles to users or service accounts.

  * Bind users to roles within a namespace (RoleBinding)
  * Or across the cluster (ClusterRoleBinding)

### 🛡️ Best Practices:

* Follow **Principle of Least Privilege** – only give necessary access.
* Avoid using `cluster-admin` role unless absolutely required.
* Regularly audit and review permissions using:

  ```bash
  kubectl auth can-i --as <user> --namespace <ns> <verb> <resource>
  ```

---

## 🌐 2. **Network Policies**

Network Policies control **traffic flow between pods** at the network level.

### 🔐 What they do:

* Define **which pods can communicate** with other pods or external endpoints.
* Applied using labels and selectors.

### ✍️ Example Policy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: my-namespace
spec:
  podSelector: {}  # apply to all pods
  policyTypes:
  - Ingress
  - Egress
```

This policy **denies all ingress and egress** traffic.

### 🛡️ Best Practices:

* Start with a **default deny-all** policy and explicitly allow necessary communication.
* Apply **namespace-level segmentation**.
* Use **labeling standards** to ensure policies apply correctly.

> ⚠️ Note: Network Policies only work if your **CNI plugin** (like Calico or Cilium) supports them.

---

## 📦 3. **Pod Security (PodSecurity Admission or PSA)**

Kubernetes Pod Security governs **what a pod is allowed to do** (e.g., privilege escalation, host access).

### 🔐 Pod Security Admission (PSA) Modes:

* **Enforce**: Blocks pods that violate the policy.
* **Audit**: Logs policy violations but does not block.
* **Warn**: Warns users but allows pod creation.

### 📋 Pod Security Levels:

* `privileged`: No restrictions.
* `baseline`: Minimal restrictions (recommended for most apps).
* `restricted`: Most secure (no privilege escalation, host namespace access, etc.)

### ✍️ Example:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: secure-ns
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/warn: baseline
```

### 🛡️ Best Practices:

* Use **`restricted`** mode for critical workloads.
* Prevent containers from:

  * Running as root (`runAsNonRoot: true`)
  * Accessing the host network or PID space
  * Using `hostPath` volumes
* Adopt **SecurityContext** and **PodSecurityContext** in pod specs.

---

## 🧩 Bonus Security Measures:

* 🔑 **Use Secrets securely**: Use tools like [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) or [Vault](https://www.vaultproject.io/) for managing secrets.
* 🔒 **Enable TLS everywhere**: Encrypt API server traffic and communication between nodes.
* 📜 **Audit Logs**: Enable audit logging to detect unauthorized activities.
* 🧪 **Regularly scan images**: Use tools like **Trivy**, **Clair**, or **Anchore** for CVE scans.
* 🛡️ **Limit access to kubelet**, etcd, and dashboard interfaces.

---
