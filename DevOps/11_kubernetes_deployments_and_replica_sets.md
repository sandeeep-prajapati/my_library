# Managing Applications in Kubernetes with Deployments and ReplicaSets

Kubernetes Deployments provide a declarative way to manage applications by controlling ReplicaSets, which in turn manage Pods. Here's how it works and a YAML example:

## Key Concepts

1. **Deployment**: Manages the desired state of your application
2. **ReplicaSet**: Ensures the specified number of pod replicas are running
3. **Rolling Updates**: Deployments support zero-downtime updates
4. **Rollbacks**: Ability to revert to previous versions

## YAML Example

Here's a complete example Deployment YAML for a simple web application:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-deployment
  labels:
    app: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: webapp-container
        image: nginx:1.19.1
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
```

## Step-by-Step Management

### 1. Create the Deployment
```bash
kubectl apply -f deployment.yaml
```

### 2. Check Deployment Status
```bash
kubectl get deployments
kubectl describe deployment webapp-deployment
```

### 3. View ReplicaSets
```bash
kubectl get replicasets
kubectl describe replicaset <replicaset-name>
```

### 4. View Pods
```bash
kubectl get pods --selector app=webapp
```

### 5. Update the Deployment (Rolling Update)
Edit the deployment.yaml file to change the image version:
```yaml
image: nginx:1.20.0
```

Then apply:
```bash
kubectl apply -f deployment.yaml
```

### 6. Check Rollout Status
```bash
kubectl rollout status deployment/webapp-deployment
```

### 7. Rollback if Needed
```bash
kubectl rollout undo deployment/webapp-deployment
```

### 8. Scale the Deployment
```bash
kubectl scale deployment webapp-deployment --replicas=5
```

## Key Benefits

1. **Self-healing**: Automatically replaces failed pods
2. **Scaling**: Easily scale up/down with simple commands
3. **Updates**: Controlled rollout of new versions
4. **Rollbacks**: Quickly revert problematic updates
