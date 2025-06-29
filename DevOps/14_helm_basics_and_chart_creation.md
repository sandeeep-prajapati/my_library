# Helm: Kubernetes Package Manager

Helm is the package manager for Kubernetes that simplifies application deployment by:
- Packaging all Kubernetes resources into a single deployable unit (chart)
- Enabling versioning and sharing of applications
- Supporting templating to customize deployments
- Managing dependencies between components

## Key Concepts

1. **Chart**: A packaged Helm application (collection of YAML templates + metadata)
2. **Release**: A deployed instance of a chart
3. **Repository**: A collection of shareable charts
4. **Values**: Customizable parameters for the chart

## Creating a Helm Chart

### 1. Initialize a new chart
```bash
helm create myapp-chart
```

This creates a directory structure:
```
myapp-chart/
  ├── Chart.yaml          # Chart metadata
  ├── values.yaml         # Default configuration values
  ├── charts/             # Subcharts/dependencies
  └── templates/          # Kubernetes resource templates
      ├── deployment.yaml
      ├── service.yaml
      ├── ingress.yaml
      └── _helpers.tpl    # Template helpers
```

### 2. Customize the Chart

**Edit Chart.yaml**:
```yaml
apiVersion: v2
name: myapp-chart
description: A Helm chart for my application
version: 0.1.0
appVersion: 1.0.0
```

**Modify templates/deployment.yaml** (example snippet):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "myapp-chart.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        ports:
        - containerPort: {{ .Values.service.port }}
```

**Update values.yaml**:
```yaml
replicaCount: 3
image:
  repository: nginx
  tag: stable
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 80
```

### 3. Package Dependencies (if any)
```bash
helm dependency update myapp-chart
```

## Deploying the Helm Chart

### 1. Install the Chart
```bash
helm install myapp-release ./myapp-chart
```

### 2. Upgrade with Custom Values
Create a custom values file `custom-values.yaml`:
```yaml
replicaCount: 5
image:
  tag: latest
```

Then upgrade:
```bash
helm upgrade myapp-release ./myapp-chart -f custom-values.yaml
```

### 3. Verify Deployment
```bash
helm list
helm status myapp-release
kubectl get pods
```

### 4. Rollback (if needed)
```bash
helm rollback myapp-release 0  # Revert to revision 0
```

## Why Helm Simplifies Kubernetes Deployments

1. **Templating**: Avoids YAML duplication with Go templating
   ```yaml
   env:
   {{- range $key, $value := .Values.envVars }}
     - name: {{ $key }}
       value: {{ $value | quote }}
   {{- end }}
   ```

2. **Value Management**: Single place to control all parameters
   ```yaml
   # values.yaml
   resources:
     limits:
       cpu: 500m
       memory: 512Mi
   ```

3. **Release Management**: Track versions and rollbacks
   ```bash
   helm history myapp-release
   ```

4. **Hooks**: Execute jobs at specific points in release lifecycle
   ```yaml
   annotations:
     "helm.sh/hook": pre-install
   ```

5. **Sharing**: Publish to chart repositories
   ```bash
   helm package myapp-chart
   helm push myapp-chart-0.1.0.tgz my-repo
   ```

## Advanced Features

- **Library Charts**: Reusable chart components
- **Chart Tests**: Validation for installed charts
- **Subcharts**: Modular application components
- **Post-renderers**: Customize manifests after templating

Helm transforms Kubernetes deployments from manual YAML management to a structured, repeatable process with version control and easy customization.