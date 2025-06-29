# Kubernetes Ingress: Managing External Access to Services

Kubernetes Ingress is an API object that provides HTTP/HTTPS routing rules to manage external access to services in a cluster. It acts as a smart traffic router, offering a more flexible alternative to NodePort or LoadBalancer services.

## How Ingress Works

1. **Ingress Resource**: You define rules in a YAML manifest that specify how to route incoming requests
2. **Ingress Controller**: A pod that watches for Ingress resources and configures the underlying load balancer/proxy
3. **External Access**: Clients connect to the Ingress endpoint which then routes to appropriate services

## Key Components

### 1. Ingress Resource (YAML Definition)
Defines the routing rules, TLS certificates, and backend services.

### 2. Ingress Controller
Actual implementation that processes the rules (common options: NGINX, Traefik, HAProxy, ALB, Istio)

### 3. Backend Services
Your regular ClusterIP services that Ingress routes to

## Example Ingress YAML

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: tls-secret
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /webapp
        pathType: Prefix
        backend:
          service:
            name: webapp-service
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

## How Ingress Helps Manage External Access

1. **Host-Based Routing**: Route traffic based on domain name
   ```yaml
   - host: api.example.com → api-service
   - host: app.example.com → webapp-service
   ```

2. **Path-Based Routing**: Route based on URL path
   ```yaml
   - path: /v1 → legacy-service
   - path: /v2 → modern-service
   ```

3. **TLS Termination**: Handle SSL certificates at the ingress level
   ```yaml
   tls:
   - hosts: [myapp.example.com]
     secretName: tls-secret
   ```

4. **Load Balancing**: Distribute traffic across service pods

5. **Name-Based Virtual Hosting**: Serve multiple websites from single IP

6. **Traffic Control**: Annotations for rate limiting, redirects, rewrites

## Benefits Over Other Approaches

| Feature          | Ingress | LoadBalancer | NodePort |
|------------------|---------|--------------|----------|
| Single IP        | ✅       | ❌ (per service) | ❌        |
| Path routing     | ✅       | ❌            | ❌        |
| Host routing     | ✅       | ❌            | ❌        |
| TLS termination  | ✅       | ✅            | ❌        |
| Cost efficiency  | ✅       | ❌            | ✅        |

## Typical Workflow

1. Deploy an Ingress Controller (once per cluster)
2. Create ClusterIP Services for your applications
3. Define Ingress resources with routing rules
4. The Ingress Controller configures itself based on your rules
5. External DNS points to the Ingress Controller's IP

## Common Ingress Controllers

- **NGINX Ingress**: Most popular, feature-rich
- **Traefik**: Dynamic configuration, good for microservices
- **AWS ALB Ingress**: Native integration with AWS ALB
- **HAProxy**: High performance
- **Istio Gateway**: Part of service mesh

Ingress provides a powerful, declarative way to manage external access while keeping your services abstracted from networking details.