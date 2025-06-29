# Setting Up Prometheus for Kubernetes Monitoring

Prometheus is the de facto standard for monitoring Kubernetes applications. Here's a comprehensive guide to setting it up and configuring key metrics and alerts.

## Installation Methods

### 1. Using Helm (Recommended)
```bash
# Add Prometheus community charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus stack (includes Grafana)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### 2. Using Prometheus Operator
```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup.yaml
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/
```

## Key Components

1. **Prometheus Server**: Time-series database and evaluation engine
2. **Alertmanager**: Handles alerts routing and deduplication
3. **Grafana**: Visualization dashboard (included in kube-prometheus-stack)
4. **Exporters**:
   - kube-state-metrics: Kubernetes object state metrics
   - node-exporter: Node-level hardware metrics
   - application-specific exporters

## Key Kubernetes Metrics to Monitor

### Cluster-Level Metrics
| Metric | Description | Alert Threshold Example |
|--------|-------------|-------------------------|
| `kube_node_status_condition` | Node health conditions | `condition="Ready",status!="true"` |
| `kube_pod_status_phase` | Pod status phases | `phase!="Running" for 5m` |
| `kube_deployment_status_replicas_unavailable` | Unavailable replicas | `> 0 for 10m` |
| `kube_node_status_allocatable_cpu_cores` | Available CPU | `(sum by (node) - sum by (node) (rate(container_cpu_usage_seconds_total[5m])) / sum by (node) (kube_node_status_allocatable_cpu_cores) > 0.9` |

### Node-Level Metrics
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `node_memory_MemAvailable_bytes` | Available memory | `< 10% total memory` |
| `node_cpu_seconds_total` | CPU usage | `> 90% utilization` |
| `node_filesystem_avail_bytes` | Disk space | `< 15% free space` |
| `node_network_receive_bytes_total` | Network traffic | Spikes detection |

### Application-Level Metrics
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `http_requests_total` | Request count | Sudden drops |
| `http_request_duration_seconds` | Latency | `99th percentile > 1s` |
| `container_memory_working_set_bytes` | Memory usage | `> 90% of limit` |
| `container_cpu_usage_seconds_total` | CPU usage | `> 80% of limit` |

## Configuring Service Monitoring

### 1. ServiceMonitor for Your Application
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: webapp-monitor
  namespace: your-app-ns
spec:
  selector:
    matchLabels:
      app: webapp
  endpoints:
  - port: web
    path: /metrics
    interval: 30s
```

### 2. PodMonitor for Stateful Applications
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: redis-monitor
spec:
  selector:
    matchLabels:
      app: redis
  podMetricsEndpoints:
  - port: metrics
```

## Alerting Rules Configuration

### 1. Custom Alert Rules
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: custom-rules
  namespace: monitoring
spec:
  groups:
  - name: node-alerts
    rules:
    - alert: HighNodeCPU
      expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100 > 90
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage on {{ $labels.instance }}"
        description: "CPU usage is {{ $value }}%"
        
  - name: application-alerts
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service) > 1
      for: 5m
      labels:
        severity: critical
```

## Key Alert Categories

1. **Resource Alerts**:
   - CPU/Memory/Disk pressure
   - Node not ready
   - Persistent volume claims full

2. **Application Alerts**:
   - High error rates (`rate(http_requests_total{status_code=~"5.."}[5m])`)
   - Increased latency
   - Dead letter queues

3. **Business Metrics**:
   - Transaction failures
   - Order processing delays
   - API rate limiting

## Accessing the Dashboards

1. **Prometheus UI**:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-operated 9090
   ```
   Access at `http://localhost:9090`

2. **Grafana**:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
   ```
   Default credentials: admin/prom-operator

## Advanced Configuration

### 1. Long-Term Storage with Thanos
```yaml
# values.yaml for kube-prometheus-stack
prometheus:
  prometheusSpec:
    thanos:
      objectStorageConfig:
        key: thanos.yaml
        name: thanos-objstore-config
```

### 2. Custom Scraping Config
```yaml
additionalScrapeConfigs:
- job_name: 'external-service'
  static_configs:
  - targets: ['external.service:9115']
```

### 3. Recording Rules
```yaml
groups:
- name: recording-rules
  rules:
  - record: cluster:node_cpu:avg_rate5m
    expr: avg by (cluster) (rate(node_cpu_seconds_total[5m]))
```

## Best Practices

1. **Labeling Strategy**: Consistent labels across services
2. **Metric Cardinality**: Avoid high-cardinality labels
3. **Alert Fatigue**: Set meaningful thresholds and durations
4. **SLO Monitoring**: Track error budgets
   ```promql
   (sum(rate(http_requests_total{status_code!~"5.."}[7d])) / sum(rate(http_requests_total[7d])))
   ```
5. **Multi-Cluster**: Use Thanos or Cortex for federation

This setup provides comprehensive visibility into your Kubernetes applications with meaningful alerts to proactively address issues before they impact users.