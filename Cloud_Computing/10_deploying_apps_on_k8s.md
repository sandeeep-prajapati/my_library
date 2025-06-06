Deploying a simple web application on Kubernetes involves several steps, including creating a container image of your application, setting up a Kubernetes cluster, and defining the necessary Kubernetes resources like deployments and services. Here’s a step-by-step guide to help you deploy a basic web application on Kubernetes.

### Step 1: Prepare Your Web Application

For this example, we’ll use a simple web application written in Python using Flask. Ensure you have the following directory structure:

```
my-web-app/
│
├── app.py         # Your Flask application
├── requirements.txt  # Dependencies for your application
└── Dockerfile     # Dockerfile for building the image
```

#### Sample Code for `app.py`

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Kubernetes!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Sample Content for `requirements.txt`

```
Flask==2.0.1
```

#### Sample Dockerfile

```Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY app.py .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the app
CMD ["python", "app.py"]
```

### Step 2: Build and Push Your Docker Image

1. **Build the Docker Image**:

   Open a terminal in the `my-web-app` directory and run:

   ```bash
   docker build -t my-web-app:latest .
   ```

2. **Run the Docker Image Locally (Optional)**:

   To test the image locally, you can run:

   ```bash
   docker run -p 5000:5000 my-web-app:latest
   ```

   Access the application at `http://localhost:5000`.

3. **Push the Image to a Container Registry**:

   If you're using Docker Hub, tag and push your image:

   ```bash
   docker tag my-web-app:latest your_dockerhub_username/my-web-app:latest
   docker push your_dockerhub_username/my-web-app:latest
   ```

### Step 3: Set Up a Kubernetes Cluster

You can use various platforms to create a Kubernetes cluster, such as:

- **Minikube**: For local development.
- **Google Kubernetes Engine (GKE)**: For Google Cloud users.
- **Amazon EKS**: For AWS users.
- **Azure Kubernetes Service (AKS)**: For Azure users.

For this guide, we will assume you're using **Minikube** for local development.

1. **Start Minikube**:

   ```bash
   minikube start
   ```

2. **Set up kubectl**:

   Ensure you have `kubectl` installed and configured to communicate with your Minikube cluster.

### Step 4: Create Kubernetes Deployment and Service

1. **Create a Deployment**:

   Create a file named `deployment.yaml` with the following content:

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: my-web-app
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: my-web-app
     template:
       metadata:
         labels:
           app: my-web-app
       spec:
         containers:
         - name: my-web-app
           image: your_dockerhub_username/my-web-app:latest
           ports:
           - containerPort: 5000
   ```

   Deploy the application:

   ```bash
   kubectl apply -f deployment.yaml
   ```

2. **Create a Service**:

   Create a file named `service.yaml` with the following content:

   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: my-web-app-service
   spec:
     type: NodePort
     selector:
       app: my-web-app
     ports:
       - port: 5000
         targetPort: 5000
         nodePort: 30000
   ```

   Deploy the service:

   ```bash
   kubectl apply -f service.yaml
   ```

### Step 5: Access the Application

After deploying your application and service, you can access it via the Minikube IP address and the assigned NodePort.

1. **Get the Minikube IP**:

   ```bash
   minikube ip
   ```

2. **Access the Application**:

   Open your web browser and navigate to:

   ```
   http://<minikube_ip>:30000
   ```

   You should see "Hello, Kubernetes!" displayed in your browser.

### Step 6: Monitor and Manage the Deployment

You can monitor your deployment using the following commands:

- **Get Deployment Status**:

   ```bash
   kubectl get deployments
   ```

- **Get Pods**:

   ```bash
   kubectl get pods
   ```

- **Get Services**:

   ```bash
   kubectl get services
   ```

### Step 7: Cleanup

To remove the deployment and service, you can run:

```bash
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
```

### Summary

You have successfully deployed a simple web application on Kubernetes using a Docker container. This guide covered creating a Docker image, setting up a Kubernetes cluster, creating deployments and services, and accessing the application. Kubernetes allows you to manage containerized applications with ease and provides features such as scaling, self-healing, and rolling updates for a seamless deployment experience.