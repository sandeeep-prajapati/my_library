# Jenkins Pipeline as Code: Declarative vs Scripted Pipelines

Jenkins Pipeline as Code is an infrastructure-as-code approach that defines CI/CD pipelines using code (typically Groovy) stored in version control, rather than configuring jobs through Jenkins' UI.

## Pipeline as Code Benefits
- **Version-controlled** pipeline definitions
- **Reproducible** builds across environments
- **Code review** for pipeline changes
- **Reusable** components via shared libraries
- **Auditable** change history

## Declarative Pipeline

The newer, simpler syntax with opinionated structure, ideal for most use cases.

### Basic Structure:
```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                // Build steps
            }
        }
        stage('Test') {
            steps {
                // Test steps
            }
        }
    }
    post {
        always {
            // Cleanup steps
        }
    }
}
```

### Complete Example:
```groovy
// Jenkinsfile (Declarative)
pipeline {
    agent {
        docker {
            image 'maven:3.8.6-jdk-11'
            args '-v $HOME/.m2:/root/.m2' // Cache Maven artifacts
        }
    }
    
    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    
    environment {
        VERSION = "1.0.${BUILD_NUMBER}"
        DOCKER_IMAGE = "my-registry/app:${VERSION}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
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
        
        stage('Docker Build') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.build(DOCKER_IMAGE)
                }
            }
        }
    }
    
    post {
        success {
            slackSend channel: '#builds',
                     message: "SUCCESS: ${JOB_NAME} #${BUILD_NUMBER}"
        }
        failure {
            slackSend channel: '#builds',
                     message: "FAILED: ${JOB_NAME} #${BUILD_NUMBER}"
        }
    }
}
```

## Scripted Pipeline

The original flexible Groovy-based syntax for advanced use cases.

### Basic Structure:
```groovy
node {
    stage('Build') {
        // Build steps
    }
    stage('Test') {
        // Test steps
    }
}
```

### Complete Example:
```groovy
// Jenkinsfile (Scripted)
def dockerImage = null

node('docker-agent') {
    checkout scm
    
    stage('Build') {
        docker.image('maven:3.8.6-jdk-11').inside('-v $HOME/.m2:/root/.m2') {
            sh 'mvn clean package'
        }
        archiveArtifacts artifacts: 'target/*.jar', fingerprint: true
    }
    
    stage('Test') {
        docker.image('maven:3.8.6-jdk-11').inside {
            sh 'mvn test'
        }
        junit 'target/surefire-reports/*.xml'
    }
    
    if (env.BRANCH_NAME == 'main') {
        stage('Docker Build') {
            dockerImage = docker.build("my-registry/app:1.0.${BUILD_NUMBER}")
        }
        
        stage('Deploy') {
            docker.withRegistry('https://my-registry', 'docker-creds') {
                dockerImage.push()
            }
            sh "kubectl set image deployment/app app=my-registry/app:1.0.${BUILD_NUMBER}"
        }
    }
}

// Post-build actions
if (currentBuild.result == 'SUCCESS') {
    slackSend channel: '#builds', 
             message: "SUCCESS: ${JOB_NAME} #${BUILD_NUMBER}"
} else {
    slackSend channel: '#builds',
             message: "FAILED: ${JOB_NAME} #${BUILD_NUMBER}"
}
```

## Key Differences

| Feature                | Declarative                          | Scripted                          |
|------------------------|--------------------------------------|-----------------------------------|
| **Syntax**             | Structured with predefined sections  | Flexible Groovy code              |
| **Learning Curve**     | Easier                               | Steeper (requires Groovy knowledge)|
| **Validation**         | Early syntax validation              | Runtime error checking            |
| **When to Use**        | Most standard CI/CD pipelines        | Complex logic/advanced workflows  |
| **Directives**         | Uses `pipeline`, `stages`, etc.      | Uses Groovy constructs            |
| **Error Handling**     | Built-in post conditions             | Manual try/catch blocks           |
| **Restartability**     | Built-in checkpoint support          | Manual implementation             |

## Best Practices for Both Styles

1. **Store in SCM**: Always keep Jenkinsfile in version control
2. **Start Simple**: Begin with declarative, switch to scripted only when needed
3. **Use Shared Libraries**: For reusable functions across pipelines
4. **Parameterize Builds**: Allow runtime configuration
   ```groovy
   parameters {
       string(name: 'DEPLOY_ENV', defaultValue: 'staging')
   }
   ```
5. **Secure Credentials**: Use Jenkins credential store
   ```groovy
   environment {
       AWS_ACCESS_KEY_ID = credentials('aws-access-key')
   }
   ```
6. **Clean Up**: Always clean workspace in post-build
   ```groovy
   post {
       always {
           cleanWs()
       }
   }
   ```

## When to Choose Which

### Use Declarative When:
- You need a simple, readable pipeline
- You want built-in syntax checking
- Your workflow fits standard CI/CD patterns
- You're new to Jenkins Pipelines

### Use Scripted When:
- You need complex logic (loops, conditionals)
- You require advanced error handling
- You're integrating with external systems
- You need dynamic pipeline generation

Both approaches can coexist - you can call scripted code from declarative pipelines using the `script` block when needed.