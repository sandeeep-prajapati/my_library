To containerize your Django project with **Docker** and **Docker Compose**, you need to follow a series of steps to create a Docker environment for both the application and its dependencies. This will allow you to run your Django app in isolated containers, making it easier to develop, deploy, and manage.

### **Step 1: Install Docker**

If you don't have Docker installed on your machine, you can follow the installation guide from the official Docker documentation for your operating system:
- [Docker Installation Guide](https://docs.docker.com/get-docker/)

### **Step 2: Create a Dockerfile**

A `Dockerfile` defines the environment for your Django application. It specifies the base image, dependencies, and steps needed to build the container.

1. In the root of your Django project (`todo_project/`), create a file named `Dockerfile`:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to avoid buffering output
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 for the Django app
EXPOSE 8000

# Run the Django development server when the container starts
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

- **Explanation**:
  - `FROM python:3.11-slim`: Specifies the base image (Python 3.11).
  - `WORKDIR /app`: Sets the working directory in the container to `/app`.
  - `COPY . /app/`: Copies the current directory’s contents into the `/app/` directory in the container.
  - `RUN pip install -r requirements.txt`: Installs the required Python packages from `requirements.txt`.
  - `CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]`: Runs the Django development server on all available IP addresses within the container (i.e., accessible from your host).

### **Step 3: Create a Requirements File**

Ensure you have a `requirements.txt` file in your project’s root directory to specify the dependencies. If you don’t have one, create it using the following command:

```bash
pip freeze > requirements.txt
```

Make sure it includes Django and any other required packages (e.g., `djangorestframework` for your API).

### **Step 4: Create a Docker Compose File**

`Docker Compose` allows you to define and run multi-container Docker applications. In your case, you’ll be running a Django container along with a PostgreSQL container for your database.

1. Create a file named `docker-compose.yml` in the root of your project (`todo_project/`):

```yaml
version: '3'

services:
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: todo_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password

  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      - db

volumes:
  postgres_data:
```

- **Explanation**:
  - `db`: Defines the PostgreSQL database service.
    - Uses the `postgres:13` image.
    - Sets environment variables (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`).
    - Persists database data in a Docker volume `postgres_data`.
  - `web`: Defines the Django web service.
    - Builds the image using the `Dockerfile` in the current directory (`build: .`).
    - Mounts the project directory as a volume, so changes to the code will reflect inside the container (`volumes: .:/app`).
    - Maps port 8000 on the container to port 8000 on the host (`ports: "8000:8000"`).
    - Depends on the `db` service, meaning the web service will start after the database service is ready.

---

### **Step 5: Update Django Settings for PostgreSQL**

In your `todo_project/settings.py`, modify the database settings to connect to PostgreSQL:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'todo_db',
        'USER': 'postgres',
        'PASSWORD': 'your_password',
        'HOST': 'db',  # The name of the service defined in docker-compose.yml
        'PORT': '5432',
    }
}
```

### **Step 6: Build and Run the Docker Containers**

1. Build the Docker images and start the containers using Docker Compose:

```bash
docker-compose up --build
```

- This command will:
  - Build the `web` (Django) and `db` (PostgreSQL) containers.
  - Start the services defined in `docker-compose.yml`.

2. After the containers are up, you should be able to access your Django app by visiting `http://localhost:8000` in your browser.

---

### **Step 7: Create Database Migrations**

Once the containers are running, open a terminal and run the following commands to set up your database:

1. Open the Django container:

```bash
docker-compose exec web bash
```

2. Run Django migrations:

```bash
python manage.py migrate
```

---

### **Step 8: Access the Django Admin**

Now you can access the Django admin interface. If you need to create a superuser to log into the admin panel:

1. Create a superuser inside the Django container:

```bash
python manage.py createsuperuser
```

2. After the superuser is created, go to `http://localhost:8000/admin` to log in.

---

### **Step 9: Stopping and Restarting the Containers**

To stop the containers, use:

```bash
docker-compose down
```

To restart the containers:

```bash
docker-compose up
```

---

### **Conclusion**

With this setup, your Django project is now containerized using Docker and Docker Compose. The backend (Django) and database (PostgreSQL) are running in separate containers, which makes it easy to scale, manage, and deploy the application. You can also easily deploy this setup to cloud services like AWS, Google Cloud, or Azure with minimal changes.