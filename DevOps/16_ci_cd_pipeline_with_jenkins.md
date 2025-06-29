# Setting Up a CI/CD Pipeline in Jenkins

Here's a complete guide to creating a basic Jenkins pipeline that compiles code and deploys it to a target environment.

## Prerequisites
- Jenkins installed (with Pipeline plugin)
- Git repository with your code
- Build tools (Maven, Gradle, npm, etc.) installed on Jenkins agents
- Deployment target access (Kubernetes, server, etc.)

## Step 1: Create a New Pipeline Job

1. In Jenkins dashboard, click **New Item**
2. Enter a name (e.g., "MyApp-CI-CD") and select **Pipeline**
3. Click **OK**

## Step 2: Configure the Pipeline

### Option A: Pipeline Script (Jenkinsfile in SCM)
1. In your project root, create a `Jenkinsfile` with the pipeline definition
2. In Jenkins job configuration:
   - Select **Pipeline script from SCM**
   - Choose your SCM (Git, GitHub, etc.)
   - Specify repository URL and credentials
   - Set branch (e.g., `main`)
   - Specify script path (`Jenkinsfile`)

### Option B: Direct Pipeline Script
1. In Jenkins job configuration:
   - Select **Pipeline script**
   - Paste the pipeline definition directly

## Basic Pipeline Example (Java/Maven to Kubernetes)

Here's a complete `Jenkinsfile` example:

```groovy
pipeline {
    agent any
    
    environment {
        // Define environment variables
        APP_NAME = "my-java-app"
        VERSION = "1.0.${BUILD_NUMBER}"
        DOCKER_IMAGE = "myregistry/${APP_NAME}:${VERSION}"
        KUBE_CONFIG = credentials('kubeconfig') // Stored in Jenkins credentials
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                url: 'https://github.com/your-repo/your-java-app.git'
            }
        }
        
        stage('Build') {
            steps {
                sh 'mvn clean package'
                archiveArtifacts artifacts: 'target/*.jar', fingerprint: true
            }
        }
        
        stage('Test') {
            steps {
                sh 'mvn test'
                junit 'target/surefire-reports/*.xml'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build(DOCKER_IMAGE)
                }
            }
        }
        
        stage('Push Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.example.com', 'dockerhub-creds') {
                        docker.image(DOCKER_IMAGE).push()
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                sh """
                    kubectl --kubeconfig=${KUBE_CONFIG} set image deployment/${APP_NAME} ${APP_NAME}=${DOCKER_IMAGE}
                """
            }
        }
    }
    
    post {
        always {
            cleanWs() // Clean workspace
        }
        success {
            slackSend channel: '#deployments',
                     message: "SUCCESS: Job ${env.JOB_NAME} build ${env.BUILD_NUMBER}"
        }
        failure {
            slackSend channel: '#deployments',
                     message: "FAILED: Job ${env.JOB_NAME} build ${env.BUILD_NUMBER}"
        }
    }
}
```

## Key Components Explained

### 1. Pipeline Structure
- `agent`: Where the pipeline runs (any available agent)
- `environment`: Shared variables for all stages
- `stages`: The main workflow steps
- `post`: Actions after pipeline completes

### 2. Core Stages
1. **Checkout**: Gets code from version control
2. **Build**: Compiles source code (Maven in this case)
3. **Test**: Runs unit tests and reports results
4. **Docker Build**: Creates container image
5. **Push Image**: Stores image in registry
6. **Deploy**: Updates Kubernetes deployment

### 3. Advanced Features
- **Credentials**: Securely stored in Jenkins (kubeconfig, Docker Hub)
- **Artifact Archiving**: Saves build outputs
- **Test Reporting**: JUnit test results visualization
- **Notifications**: Slack integration

## Setting Up Required Plugins

Ensure these Jenkins plugins are installed:
- Pipeline
- Docker Pipeline
- Kubernetes CLI
- Git
- JUnit
- Slack Notification
- Credentials Binding

## Best Practices

1. **Use Jenkinsfile in SCM**: Store pipeline definition with your code
2. **Secure Credentials**: Never hardcode secrets - use Jenkins credentials store
3. **Parallelize Stages**: Run independent stages in parallel
4. **Implement Rollbacks**: Add stage to revert failed deployments
5. **Parameterize Builds**: Allow runtime inputs:
   ```groovy
   parameters {
       choice(name: 'DEPLOY_ENV', choices: ['dev', 'staging', 'prod'], description: 'Target environment')
       booleanParam(name: 'RUN_TESTS', defaultValue: true, description: 'Run all tests?')
   }
   ```
6. **Use Shared Libraries**: Reusable pipeline components across projects

## Visualizing the Pipeline

Jenkins provides several visualization tools:
- **Blue Ocean**: Modern UI for pipeline visualization
- **Stage View**: Traditional stage progress view
- **Build History**: Timeline of past executions

This pipeline gives you a complete CI/CD workflow from code commit to production deployment, with testing and artifact management built in. You can extend it with additional stages like security scanning, performance testing, or approval gates for production deployments.