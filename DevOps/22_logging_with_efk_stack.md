# Setting Up EFK Stack for Kubernetes Logging

The EFK (Elasticsearch, Fluentd, Kibana) stack is a powerful solution for centralized logging in Kubernetes environments. Here's a complete guide to deploying and configuring it:

## Prerequisites
- Kubernetes cluster (v1.19+ recommended)
- `kubectl` configured with cluster access
- Helm v3+ installed
- Minimum 4GB RAM available in cluster

## Deployment Methods

### Method 1: Using Helm (Recommended)

1. **Add Helm Repositories**:
   ```bash
   helm repo add elastic https://helm.elastic.co
   helm repo add fluent https://fluent.github.io/helm-charts
   helm repo update
   ```

2. **Install Elasticsearch**:
   ```bash
   helm install elasticsearch elastic/elasticsearch \
     --namespace logging \
     --create-namespace \
     --set replicas=3 \
     --set minimumMasterNodes=2 \
     --set resources.requests.memory=4Gi \
     --set resources.limits.memory=8Gi
   ```

3. **Install Fluentd**:
   ```bash
   helm install fluentd fluent/fluentd \
     --namespace logging \
     --set elasticsearch.host=elasticsearch-master.logging.svc.cluster.local \
     --set elasticsearch.port=9200 \
     --set resources.requests.memory=512Mi \
     --set resources.limits.memory=1Gi
   ```

4. **Install Kibana**:
   ```bash
   helm install kibana elastic/kibana \
     --namespace logging \
     --set service.type=LoadBalancer \
     --set elasticsearchHosts=http://elasticsearch-master.logging.svc.cluster.local:9200
   ```

### Method 2: Manual YAML Deployment

1. **Create Namespace**:
   ```bash
   kubectl create namespace logging
   ```

2. **Deploy Elasticsearch StatefulSet** (`elasticsearch.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: StatefulSet
   metadata:
     name: elasticsearch
     namespace: logging
   spec:
     serviceName: elasticsearch
     replicas: 3
     selector:
       matchLabels:
         app: elasticsearch
     template:
       metadata:
         labels:
           app: elasticsearch
       spec:
         containers:
         - name: elasticsearch
           image: docker.elastic.co/elasticsearch/elasticsearch:7.17.3
           ports:
           - containerPort: 9200
             name: http
           - containerPort: 9300
             name: transport
           volumeMounts:
           - name: data
             mountPath: /usr/share/elasticsearch/data
           env:
           - name: discovery.type
             value: single-node # For production, use "zen" with proper config
           - name: ES_JAVA_OPTS
             value: "-Xms2g -Xmx2g"
         volumes:
         - name: data
           emptyDir: {}
   ```

3. **Deploy Fluentd DaemonSet** (`fluentd.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: DaemonSet
   metadata:
     name: fluentd
     namespace: logging
   spec:
     selector:
       matchLabels:
         app: fluentd
     template:
       metadata:
         labels:
           app: fluentd
       spec:
         containers:
         - name: fluentd
           image: fluent/fluentd-kubernetes-daemonset:v1.14.6-debian-elasticsearch7-1.0
           env:
           - name: FLUENT_ELASTICSEARCH_HOST
             value: "elasticsearch.logging.svc.cluster.local"
           - name: FLUENT_ELASTICSEARCH_PORT
             value: "9200"
           volumeMounts:
           - name: varlog
             mountPath: /var/log
           - name: varlibdockercontainers
             mountPath: /var/lib/docker/containers
             readOnly: true
         volumes:
         - name: varlog
           hostPath:
             path: /var/log
         - name: varlibdockercontainers
           hostPath:
             path: /var/lib/docker/containers
   ```

4. **Deploy Kibana Deployment** (`kibana.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: kibana
     namespace: logging
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: kibana
     template:
       metadata:
         labels:
           app: kibana
       spec:
         containers:
         - name: kibana
           image: docker.elastic.co/kibana/kibana:7.17.3
           ports:
           - containerPort: 5601
           env:
           - name: ELASTICSEARCH_HOSTS
             value: "http://elasticsearch.logging.svc.cluster.local:9200"
   ```

## Configuration Details

### Fluentd Configuration (Customize as needed)

Create a ConfigMap for custom Fluentd configuration (`fluentd-config.yaml`):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      logstash_format true
      logstash_prefix fluentd
      include_tag_key true
      type_name _doc
    </match>
```

### Elasticsearch Resource Requirements

For production environments, consider these settings:
```yaml
resources:
  requests:
    cpu: "1000m"
    memory: "4Gi"
  limits:
    cpu: "2000m"
    memory: "8Gi"
```

## Verification and Access

1. **Check Pod Status**:
   ```bash
   kubectl get pods -n logging
   ```

2. **Access Kibana**:
   ```bash
   kubectl port-forward svc/kibana-kibana 5601:5601 -n logging
   ```
   Open `http://localhost:5601`

3. **Verify Logs in Elasticsearch**:
   ```bash
   kubectl exec -it elasticsearch-0 -n logging -- curl http://localhost:9200/_cat/indices?v
   ```

## Production Considerations

1. **Persistent Storage**:
   ```yaml
   volumeClaimTemplates:
   - metadata:
       name: data
     spec:
       accessModes: [ "ReadWriteOnce" ]
       storageClassName: "standard"
       resources:
         requests:
           storage: 50Gi
   ```

2. **Security**:
   - Enable Elasticsearch security features (x-pack)
   - Use network policies to restrict access
   - Implement TLS between components

3. **Log Retention**:
   Configure Index Lifecycle Management (ILM) policies in Elasticsearch

4. **Multi-Line Logs**:
   Add Fluentd multiline parsing for stack traces:
   ```xml
   <filter kubernetes.**>
     @type concat
     key log
     multiline_start_regexp /^\d{4}-\d{2}-\d{2}/
   </filter>
   ```

5. **Custom Log Processing**:
   ```xml
   <filter app.**>
     @type record_transformer
     <record>
       service_name ${record.dig("kubernetes", "labels", "app")}
     </record>
   </filter>
   ```

## Troubleshooting

1. **No Logs in Kibana**:
   - Check Fluentd logs: `kubectl logs fluentd-xxxx -n logging`
   - Verify Elasticsearch connection from Fluentd
   - Check index patterns in Kibana

2. **Performance Issues**:
   - Increase Fluentd buffer size
   - Scale Elasticsearch nodes
   - Optimize Elasticsearch shard count

This EFK stack setup provides a robust logging solution that collects, processes, stores, and visualizes logs from all Kubernetes pods and nodes. The Helm method is recommended for most deployments as it simplifies maintenance and upgrades.