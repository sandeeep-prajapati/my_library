# Kubernetes ConfigMaps and Secrets: Managing Configuration Data

ConfigMaps and Secrets are Kubernetes objects used to decouple configuration data from container images, making applications more portable and secure.

## ConfigMaps

**Purpose**: Store non-sensitive configuration data in key-value pairs.

### Use Cases for ConfigMaps:
1. Environment variables for applications
2. Configuration files (e.g., nginx.conf, application.properties)
3. Command-line arguments
4. Any non-sensitive app configuration

### Example ConfigMap YAML:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # Key-value pairs
  APP_COLOR: blue
  APP_MODE: prod
  
  # File-like configuration
  nginx.conf: |
    server {
      listen 80;
      server_name localhost;
      location / {
        proxy_pass http://webapp-service;
      }
    }
```

### Using ConfigMap in a Pod:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: webapp-pod
spec:
  containers:
  - name: webapp
    image: nginx
    envFrom:
    - configMapRef:
        name: app-config  # Import all key-value pairs as env vars
    volumeMounts:
    - name: config-volume
      mountPath: /etc/nginx
  volumes:
  - name: config-volume
    configMap:
      name: app-config   # Mount specific config files
```

## Secrets

**Purpose**: Store sensitive data like passwords, API keys, and TLS certificates in an encrypted form (base64 encoded, not encrypted by default - consider external secret management for production).

### Use Cases for Secrets:
1. Database credentials
2. API tokens/keys
3. TLS certificates
4. SSH keys
5. Any sensitive configuration

### Example Secret YAML:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque  # Default type for arbitrary data
data:
  # Base64 encoded values (echo -n "value" | base64)
  DB_USER: cm9vdA==
  DB_PASSWORD: cGFzc3dvcmQxMjM=
```

### Using Secret in a Pod:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: db-pod
spec:
  containers:
  - name: database
    image: mysql
    env:
    - name: MYSQL_ROOT_PASSWORD  # Environment variable name
      valueFrom:
        secretKeyRef:
          name: db-secret        # Secret name
          key: DB_PASSWORD      # Key in secret
    volumeMounts:
    - name: cert-volume
      mountPath: /etc/mysql/certs
  volumes:
  - name: cert-volume
    secret:
      secretName: tls-secret     # Mount secret as files
```

## Key Differences

| Feature          | ConfigMap               | Secret                     |
|------------------|-------------------------|----------------------------|
| Data type        | Plain text              | Base64 encoded             |
| Security         | Non-sensitive data      | Sensitive data             |
| Storage          | etcd (unencrypted)      | etcd (base64, not encrypted) |
| Use case         | Configuration files     | Credentials, certificates  |
| Size limitation  | 1MB per ConfigMap       | 1MB per Secret             |

## Best Practices

1. **For Secrets**:
   - Use third-party secret managers (AWS Secrets Manager, HashiCorp Vault) for production
   - Enable etcd encryption at rest
   - Use Kubernetes RBAC to restrict access
   - Consider using SealedSecrets for GitOps workflows

2. **For ConfigMaps**:
   - Use immutable ConfigMaps (`immutable: true`) for better performance
   - Group related configurations together
   - Use annotations for metadata/versioning

3. **For Both**:
   - Update references when configs change (may require pod restart)
   - Use rolling updates when configuration changes
   - Consider using ConfigMap/Secret generators with Kustomize

These mechanisms enable proper separation of configuration from application code, making your Kubernetes applications more secure and portable across environments.