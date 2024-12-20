To develop a full-stack To-Do app using **Django** for both the backend and the frontend, we’ll need to follow a few steps to set up the app’s structure and functionality. Here's a step-by-step guide:

---

### **Step 1: Set Up the Django Project**

1. **Install Django:**
   If you haven't already, install Django:

   ```bash
   pip install django
   ```

2. **Create a New Django Project:**

   ```bash
   django-admin startproject todo_project
   cd todo_project
   ```

3. **Create a Django App:**

   Create an app to handle the To-Do logic:

   ```bash
   python manage.py startapp todo
   ```

4. **Add the App to `INSTALLED_APPS`:**

   In `todo_project/settings.py`, add `'todo'` to the `INSTALLED_APPS` list:

   ```python
   INSTALLED_APPS = [
       ...
       'todo',
   ]
   ```

---

### **Step 2: Set Up the To-Do Model**

In the `todo` app, define a model for the To-Do items in `todo/models.py`:

```python
from django.db import models

class Todo(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

- `title`: Name of the task.
- `description`: Optional description for the task.
- `completed`: Boolean to mark whether the task is completed.
- `created_at`: Automatically stores the creation time.

---

### **Step 3: Create Database Migrations**

Run migrations to create the database schema for the `Todo` model.

```bash
python manage.py makemigrations
python manage.py migrate
```

---

### **Step 4: Set Up Django REST Framework**

To build a REST API for the To-Do app, install Django REST Framework:

```bash
pip install djangorestframework
```

Next, add `'rest_framework'` to `INSTALLED_APPS` in `settings.py`:

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
]
```

Create a `serializers.py` file in the `todo` app to serialize the `Todo` model.

```python
# todo/serializers.py
from rest_framework import serializers
from .models import Todo

class TodoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = '__all__'
```

---

### **Step 5: Create API Views**

Now, create API views for CRUD operations on the To-Do items.

In `todo/views.py`:

```python
from rest_framework import generics
from .models import Todo
from .serializers import TodoSerializer

class TodoListCreate(generics.ListCreateAPIView):
    queryset = Todo.objects.all()
    serializer_class = TodoSerializer

class TodoDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Todo.objects.all()
    serializer_class = TodoSerializer
```

- `TodoListCreate`: Handles listing all tasks and creating new ones.
- `TodoDetail`: Handles retrieving, updating, and deleting tasks.

---

### **Step 6: Set Up URLs for API Endpoints**

In `todo/urls.py`, define the API endpoints:

```python
# todo/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('todos/', views.TodoListCreate.as_view(), name='todo-list-create'),
    path('todos/<int:pk>/', views.TodoDetail.as_view(), name='todo-detail'),
]
```

Include the `todo` URLs in the main project’s `urls.py`:

```python
# todo_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('todo.urls')),  # Include todo app URLs
]
```

---

### **Step 7: Set Up the Frontend (HTML + JavaScript)**

Create a basic frontend to interact with the Django API.

1. **Create the HTML template**:

   In `todo/templates/todo/index.html`, create a basic structure for the To-Do app:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>To-Do App</title>
</head>
<body>
    <h1>To-Do List</h1>
    <ul id="todo-list"></ul>

    <h2>Add a new task</h2>
    <input type="text" id="todo-title" placeholder="Task title">
    <textarea id="todo-description" placeholder="Task description"></textarea>
    <button onclick="createTodo()">Add Task</button>

    <script>
        const apiUrl = '/api/todos/';

        // Fetch all todos and display them
        function fetchTodos() {
            fetch(apiUrl)
                .then(response => response.json())
                .then(data => {
                    const todoList = document.getElementById('todo-list');
                    todoList.innerHTML = '';
                    data.forEach(todo => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <strong>${todo.title}</strong><br>
                            ${todo.description}<br>
                            Completed: ${todo.completed}<br>
                            <button onclick="toggleCompletion(${todo.id}, ${todo.completed})">
                                ${todo.completed ? 'Mark as Incomplete' : 'Mark as Completed'}
                            </button>
                            <button onclick="deleteTodo(${todo.id})">Delete</button>
                        `;
                        todoList.appendChild(li);
                    });
                });
        }

        // Create a new todo
        function createTodo() {
            const title = document.getElementById('todo-title').value;
            const description = document.getElementById('todo-description').value;
            const todoData = { title, description, completed: false };

            fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(todoData)
            })
            .then(response => response.json())
            .then(() => {
                fetchTodos();
            });
        }

        // Toggle the completion status of a todo
        function toggleCompletion(id, currentStatus) {
            const updatedTodo = { completed: !currentStatus };
            fetch(`/api/todos/${id}/`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(updatedTodo)
            })
            .then(() => {
                fetchTodos();
            });
        }

        // Delete a todo
        function deleteTodo(id) {
            fetch(`/api/todos/${id}/`, {
                method: 'DELETE',
            })
            .then(() => {
                fetchTodos();
            });
        }

        // Initial fetch of todos when the page loads
        fetchTodos();
    </script>
</body>
</html>
```

This HTML file provides a basic interface for interacting with the Django backend. It uses JavaScript to:
- Fetch the list of To-Do items.
- Create new To-Do items.
- Toggle completion status.
- Delete To-Do items.

---

### **Step 8: Run the Development Server**

Run the Django development server to test the app:

```bash
python manage.py runserver
```

- Visit `http://127.0.0.1:8000/` to see the To-Do list.
- Use the API endpoints `/api/todos/` to interact with the backend.

---

### **Conclusion**

With this setup, you have a full-stack Django To-Do app, which includes:

1. **Backend**: Django REST API to handle CRUD operations on To-Do items.
2. **Frontend**: Basic HTML/JavaScript to interact with the API.

This setup provides a foundational structure for a full-stack app with Django, and you can enhance it further by adding user authentication, advanced features like due dates, and deploying it to production.