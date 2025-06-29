# Helm Templates, Values, and Releases Explained with Best Practices

## Helm Core Components

### 1. Templates
- **Purpose**: Go-templated Kubernetes manifests that become actual resources when deployed
- **Location**: `templates/` directory in a chart
- **Features**:
  - Use Go template language with Sprig functions
  - Access values from `values.yaml` via `{{ .Values.param }}`
  - Include helper functions from `_helpers.tpl`

**Example Deployment Template** (`templates/deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mychart.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        {{- with .Values.resources }}
        resources:
          {{- toYaml . | nindent 10 }}
        {{- end }}
```

### 2. Values
- **Purpose**: Configuration parameters that customize the chart
- **Sources** (in order of precedence):
  1. Command-line (`--set` flags)
  2. Custom values files (`-f myvalues.yaml`)
  3. Default `values.yaml` in chart
  4. Parent chart's `values.yaml` (for subcharts)

**Example values.yaml**:
```yaml
replicaCount: 3
image:
  repository: nginx
  tag: stable
  pullPolicy: IfNotPresent
resources:
  limits:
    cpu: 200m
    memory: 256Mi
```

### 3. Releases
- **Definition**: An instance of a chart running in a Kubernetes cluster
- **Management**:
  - Each `helm install` creates a new release
  - Releases are versioned (revisions)
  - Stored as Secrets in the cluster by default

**Release Lifecycle**:
```
helm install → helm upgrade → helm rollback → helm uninstall
```

## Helm Chart Best Practices

### 1. Structure and Organization
```
mychart/
├── Chart.yaml          # Required: Chart metadata
├── values.yaml        # Required: Default configuration
├── charts/            # Optional: Subcharts/dependencies
├── templates/         # Required: Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── _helpers.tpl   # Template helpers
│   └── tests/        # Test pods
└── README.md          # Documentation
```

### 2. Templating Practices
- **Use named templates** (`_helpers.tpl`) for reusable components:
  ```tpl
  {{- define "mychart.labels" -}}
  app.kubernetes.io/name: {{ .Chart.Name }}
  app.kubernetes.io/instance: {{ .Release.Name }}
  {{- end }}
  ```
  
- **Add input validation**:
  ```yaml
  {{- if not .Values.image.tag }}
  {{- fail "image.tag is required" }}
  {{- end }}
  ```

- **Use `include` over `template`** (preserves scope):
  ```yaml
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
  ```

### 3. Values Management
- **Document all values** in `values.yaml` with comments:
  ```yaml
  # replicaCount: Number of application instances
  replicaCount: 1
  ```
  
- **Group related values** hierarchically:
  ```yaml
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 5
  ```

- **Provide sensible defaults** but allow overrides:
  ```yaml
  service:
    type: ClusterIP  # LoadBalancer for production
    port: 80
  ```

### 4. Security Practices
- **Never put secrets in values.yaml** - Use Kubernetes Secrets instead
- **Use `helm lint`** to validate charts
- **Make charts immutable** when possible:
  ```yaml
  apiVersion: v2
  name: mychart
  description: My immutable chart
  type: application
  ```

### 5. Release Management
- **Use semantic versioning** for charts:
  ```yaml
  version: 1.2.3  # MAJOR.MINOR.PATCH
  ```
  
- **Implement upgrade hooks** for migrations:
  ```yaml
  annotations:
    "helm.sh/hook": pre-upgrade
  ```

- **Test with --dry-run** before actual deployment:
  ```bash
  helm install myapp ./mychart --dry-run --debug
  ```

### 6. Advanced Techniques
- **Conditional templates**:
  ```yaml
  {{- if .Values.ingress.enabled }}
  apiVersion: networking.k8s.io/v1
  kind: Ingress
  ...
  {{- end }}
  ```

- **Range over values**:
  ```yaml
  env:
  {{- range $key, $value := .Values.envVars }}
    - name: {{ $key }}
      value: {{ $value | quote }}
  {{- end }}
  ```

- **Template partials** for complex objects:
  ```tpl
  {{- define "mychart.resources" }}
  {{- if .Values.resources }}
  resources:
    {{- toYaml .Values.resources | nindent 4 }}
  {{- end }}
  {{- end }}
  ```

## Practical Example Workflow

1. **Create a new chart**:
   ```bash
   helm create myapp && rm -rf myapp/templates/*
   ```

2. **Add templates** (deployment.yaml, service.yaml etc.)

3. **Define values** in values.yaml

4. **Package and deploy**:
   ```bash
   helm package myapp
   helm install myapp ./myapp-0.1.0.tgz --values prod-values.yaml
   ```

5. **Upgrade**:
   ```bash
   helm upgrade myapp ./myapp --set replicaCount=5
   ```

Helm's templating system combined with these best practices enables you to create maintainable, reusable, and production-grade Kubernetes deployments that can be customized for different environments while keeping your manifests DRY (Don't Repeat Yourself).