# Jenkins + Docker Integration: Automated Containerized CI/CD

Here's a complete practical guide to integrating Jenkins with Docker for automating container builds and deployments, including a working example.

## Prerequisites
- Jenkins server with Docker plugins installed
- Docker installed on Jenkins agents (or Jenkins server)
- Docker Hub/registry credentials configured in Jenkins
- Kubernetes cluster (for deployment, optional)

## Step 1: Configure Jenkins for Docker

1. **Install required plugins**:
   - Docker
   - Docker Pipeline
   - Docker Build Step

2. **Add Docker credentials** in Jenkins:
   - Go to **Credentials** → **System** → **Global credentials**
   - Add username/password for your Docker registry

## Step 2: Create a Docker-Enabled Pipeline

Here's a complete `Jenkinsfile` example that:
1. Builds a Node.js application
2. Creates a Docker image
3. Pushes to Docker Hub
4. Deploys to Kubernetes

```groovy
pipeline {
    agent {
        docker {
            image 'node:14-alpine'
            args '-u root:root' // Needed for file permissions
            reuseNode true
        }
    }

    environment {
        APP_NAME = "nodejs-app"
        DOCKER_IMAGE = "your-dockerhub/${APP_NAME}:${BUILD_NUMBER}"
        KUBE_NAMESPACE = "default"
        // Credentials from Jenkins
        DOCKER_CREDS = credentials('dockerhub-creds')
        KUBECONFIG = credentials('kubeconfig')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'npm install'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'npm test'
            }
        }

        stage('Build Docker Image') {
            agent any // Needs Docker daemon
            steps {
                script {
                    docker.build(DOCKER_IMAGE)
                }
            }
        }

        stage('Push Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKER_CREDS) {
                        docker.image(DOCKER_IMAGE).push()
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh """
                    kubectl --kubeconfig=${KUBECONFIG} -n ${KUBE_NAMESPACE} \
                    set image deployment/${APP_NAME} ${APP_NAME}=${DOCKER_IMAGE}
                """
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            slackSend(color: 'good', message: "SUCCESS: ${JOB_NAME} #${BUILD_NUMBER}")
        }
        failure {
            slackSend(color: 'danger', message: "FAILED: ${JOB_NAME} #${BUILD_NUMBER}")
        }
    }
}
```

## Key Integration Points Explained

### 1. Docker Build Agent
```groovy
agent {
    docker {
        image 'node:14-alpine' // Uses Node.js container for build
        args '-u root:root'
    }
}
```
- Runs build stages inside a containerized environment
- Ensures consistent build tools across executions

### 2. Docker Image Building
```groovy
script {
    docker.build(DOCKER_IMAGE)
}
```
- Requires a `Dockerfile` in your project root
- Example `Dockerfile` for Node.js app:
```dockerfile
FROM node:14-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### 3. Secure Registry Push
```groovy
docker.withRegistry('https://registry.hub.docker.com', DOCKER_CREDS) {
    docker.image(DOCKER_IMAGE).push()
}
```
- Uses Jenkins-stored credentials
- Supports private registries (Artifactory, ECR, GCR, etc.)

### 4. Kubernetes Deployment
```groovy
sh """
    kubectl set image deployment/${APP_NAME} ${APP_NAME}=${DOCKER_IMAGE}
"""
```
- Assumes existing Kubernetes deployment
- Alternative: `kubectl apply -f k8s-deployment.yaml`

## Advanced Integration Patterns

### 1. Multi-Architecture Builds
```groovy
stage('Build Multi-Arch Image') {
    steps {
        sh """
            docker buildx build --platform linux/amd64,linux/arm64 \
            -t ${DOCKER_IMAGE} --push .
        """
    }
}
```

### 2. Docker Compose Testing
```groovy
stage('Integration Tests') {
    steps {
        sh 'docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit'
    }
}
```

### 3. Parameterized Builds
```groovy
parameters {
    choice(name: 'ENVIRONMENT', choices: ['dev', 'staging', 'prod'])
    string(name: 'IMAGE_TAG', defaultValue: 'latest')
}
```

### 4. Docker Content Trust
```groovy
environment {
    DOCKER_CONTENT_TRUST = '1'
}
```

## Best Practices

1. **Use Docker-in-Docker (DinD)** for isolated builds:
   ```groovy
   agent {
       docker {
           image 'docker:dind'
           args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
       }
   }
   ```

2. **Cache dependencies** between builds:
   ```groovy
   stage('Install Dependencies') {
       steps {
           sh '''
               docker run -v ${PWD}:/app -v /app/node_modules \
               node:14-alpine npm install
           '''
       }
   }
   ```

3. **Scan images for vulnerabilities**:
   ```groovy
   stage('Security Scan') {
       steps {
           sh 'docker scan ${DOCKER_IMAGE}'
       }
   }
   ```

4. **Clean up old images**:
   ```groovy
   post {
       always {
           sh 'docker system prune -f || true'
       }
   }
   ```

## Monitoring the Pipeline

1. **Blue Ocean View**: Visual pipeline progress
2. **Console Output**: Detailed build logs
3. **Docker Hub**: Verify pushed images
4. **Kubernetes**: Check deployment status
   ```bash
   kubectl get pods -w
   kubectl describe deployment/nodejs-app
   ```

This integration provides a complete workflow from code commit to production deployment, leveraging Docker for consistent builds and environments throughout the pipeline.