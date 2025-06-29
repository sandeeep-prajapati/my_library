# Configuring GitHub Webhooks for Jenkins Auto-Triggered Builds

Here's a complete step-by-step guide to setting up GitHub webhooks that automatically trigger Jenkins builds when code is pushed:

## Prerequisites
- Jenkins server accessible via HTTPS (GitHub requires HTTPS for webhooks)
- GitHub repository admin access
- Jenkins GitHub Plugin installed (`GitHub plugin` and `GitHub API plugin`)

## Step 1: Configure Jenkins for GitHub Integration

### 1. Install Required Plugins
- Manage Jenkins → Manage Plugins → Available
- Install:
  - GitHub plugin
  - GitHub API plugin
  - (Optional) Pipeline: GitHub Plugin

### 2. Set Up GitHub Server in Jenkins
1. Go to **Manage Jenkins → Configure System**
2. Find **GitHub** section
3. Add GitHub Server:
   - Name: `GitHub`
   - API URL: `https://api.github.com`
4. Add credentials:
   - Kind: **Secret text**
   - Secret: [GitHub personal access token] (with `repo` scope)
   - ID: `github-token`

## Step 2: Configure the Jenkins Job

### For Freestyle Projects:
1. Create/Edit job → **Configure**
2. Under **Source Code Management**:
   - Select Git
   - Enter Repository URL: `https://github.com/your-username/your-repo.git`
3. Under **Build Triggers**:
   - Check **GitHub hook trigger for GITScm polling**

### For Pipeline Projects:
1. Use this trigger definition in your `Jenkinsfile`:
```groovy
triggers {
    githubPush()
}
```
Or for more control:
```groovy
triggers {
    GitHubPushTrigger {
        branchesToBuild = ['main', 'develop']
    }
}
```

## Step 3: Set Up GitHub Webhook

1. Go to your GitHub repository → **Settings → Webhooks → Add webhook**
2. Configure the webhook:
   - Payload URL: `https://your-jenkins-server/github-webhook/`
   - Content type: `application/json`
   - Secret: [Optional] Add a secret that matches Jenkins configuration
   - Which events: 
     - For push events: **Just the push event**
     - For all events: **Send me everything**
3. Click **Add webhook**

## Step 4: Verify the Connection

1. In GitHub, check the webhook's recent deliveries
2. In Jenkins, check:
   - **Manage Jenkins → System Log** for GitHub plugin logs
   - Job's **Build History** for triggered builds

## Troubleshooting Tips

### Common Issues and Fixes:
1. **Webhook not triggering**:
   - Verify Jenkins URL is accessible from GitHub
   - Check Jenkins logs (`/var/log/jenkins/jenkins.log`)
   - Test with `curl -X POST https://your-jenkins-server/github-webhook/`

2. **403 Forbidden errors**:
   - Ensure GitHub token has proper permissions
   - Verify webhook secret matches (if used)

3. **Builds not starting**:
   - Check "GitHub hook trigger" is enabled in job config
   - Verify SCM polling is disabled (conflicts with webhooks)

## Advanced Configuration

### 1. Filtering Specific Branches
In your `Jenkinsfile`:
```groovy
triggers {
    githubPush()
}
pipeline {
    agent any
    stages {
        stage('Build') {
            when {
                branch 'main' // Only build on main branch pushes
            }
            steps {
                sh 'make build'
            }
        }
    }
}
```

### 2. Using Webhook Secrets
1. Generate a secret:
   ```bash
   openssl rand -hex 20
   ```
2. Add to Jenkins GitHub server config
3. Use same secret in GitHub webhook

### 3. Multibranch Pipelines
1. Create **New Item → Multibranch Pipeline**
2. Configure branch sources to auto-discover branches/PRs
3. Webhooks will automatically trigger builds for all branches

## Security Best Practices

1. **Use HTTPS**: GitHub requires HTTPS for webhooks
2. **Restrict IPs**: Configure GitHub webhook to only allow Jenkins server IP
3. **Use Secrets**: Always use webhook secrets for verification
4. **Limit Permissions**: GitHub token only needs `repo` scope
5. **Rotate Secrets**: Periodically update webhook and API tokens

This setup creates a fully automated CI pipeline where any push to GitHub triggers the corresponding Jenkins job immediately, enabling true continuous integration.