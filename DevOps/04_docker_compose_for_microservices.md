# **Docker Compose: Managing Multi-Container Applications**

Docker Compose is a tool for **defining and running multi-container Docker applications** using a declarative YAML file (`docker-compose.yml`). It simplifies orchestrating interconnected services (e.g., web apps, databases, caching) with a single command.

---

## **Key Features of Docker Compose**
1. **Single-Command Management**  
   - Start, stop, and rebuild all services with `docker compose up` / `down`.
2. **Service Dependencies**  
   - Define networks, volumes, and startup order.
3. **Environment Variables & Configs**  
   - Centralize configurations (e.g., database credentials).
4. **Development & Testing**  
   - Replicate production environments locally.

---

## **Docker Compose vs. Plain Docker**
| Feature          | Docker (Single Container) | Docker Compose (Multi-Container) |
|------------------|--------------------------|---------------------------------|
| **Orchestration** | Manual (`docker run`)    | Declarative YAML                |
| **Networking**   | Manual `--network`       | Auto-created networks           |
| **Scaling**      | Difficult                | Limited (use Kubernetes for prod) |

---

## **Practical Example: Flask + Redis + PostgreSQL**
### **Project Structure**
```
myapp/
├── app.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml  # The Compose file
```

### **1. `Dockerfile` (Flask App)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### **2. `docker-compose.yml`**
```yaml
version: '3.8'

services:
  web:
    build: .  # Builds from Dockerfile
    ports:
      - "5000:5000"
    environment:
      - REDIS_HOST=redis
      - POSTGRES_HOST=db
    depends_on:
      - redis
      - db

  redis:
    image: redis:alpine  # Uses prebuilt image
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### **3. `app.py` (Flask + Redis + PostgreSQL)**
```python
from flask import Flask
import redis
import psycopg2
import os

app = Flask(__name__)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379)
postgres_conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    user="myuser",
    password="mypassword",
    dbname="mydb"
)

@app.route("/")
def hello():
    redis_client.incr("hits")
    hits = redis_client.get("hits").decode()
    return f"Hello! This page has been viewed {hits} times."

if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

---

## **Running the Application**
### **1. Start All Services**
```bash
docker compose up -d  # Detached mode
```
- Builds `web` from `Dockerfile`.
- Pulls `redis` and `postgres` images.
- Creates networks and volumes automatically.

### **2. Check Running Containers**
```bash
docker compose ps
```
Output:
```
NAME                COMMAND                  STATUS          PORTS
myapp-web-1         "python app.py"          Up              0.0.0.0:5000->5000/tcp
myapp-redis-1       "docker-entrypoint.s…"   Up              0.0.0.0:6379->6379/tcp
myapp-db-1          "docker-entrypoint.s…"   Up              5432/tcp
```

### **3. Test the Application**
```bash
curl http://localhost:5000
```
Output:
```
Hello! This page has been viewed 1 times.
```

### **4. Stop All Services**
```bash
docker compose down
```
- Removes containers but retains volumes (data persists).

---

## **Key Docker Compose Commands**
| Command | Description |
|---------|-------------|
| `docker compose up` | Start services |
| `docker compose down` | Stop and remove containers |
| `docker compose logs` | View logs |
| `docker compose ps` | List running services |
| `docker compose build` | Rebuild images |

---

## **Best Practices**
1. **Use Named Volumes** (as shown) for persistent data (DBs, Redis).
2. **Environment Variables** for secrets (use `.env` file or Docker secrets).
3. **`depends_on`** ensures services start in order (but doesn’t wait for readiness).
4. **Separate `docker-compose.override.yml`** for dev vs. prod configurations.

---

## **When to Use Docker Compose**
- **Local development** (mimicking production).
- **CI/CD testing** (spinning up dependencies like DBs).
- **Microservices prototyping** (before Kubernetes).

For **production scaling**, consider **Kubernetes** or **Docker Swarm**.

---

## **Final Thoughts**
Docker Compose simplifies **multi-container apps** by replacing manual `docker run` commands with a single YAML file. The example above demonstrates a **Flask app with Redis (caching) and PostgreSQL (database)**—a common real-world use case.  

🚀 **Now you can orchestrate containers like a pro!**