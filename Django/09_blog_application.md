### **Developing a Simple Blog App with CRUD Functionality in Django**

---

### **Step 1: Create a New App**
Run the following command to create a new app:  
```bash
python manage.py startapp blog
```

Add `blog` to `INSTALLED_APPS` in `settings.py`:  
```python
INSTALLED_APPS = [
    ...,
    'blog',
]
```

---

### **Step 2: Define the Blog Post Model**
In `blog/models.py`, create a model for blog posts:  
```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
```

---

### **Step 3: Create and Apply Migrations**
Run the following commands to create and apply the migrations:  
```bash
python manage.py makemigrations
python manage.py migrate
```

---

### **Step 4: Create Views for CRUD Operations**
In `blog/views.py`, create views for creating, reading, updating, and deleting posts:  
```python
from django.shortcuts import render, get_object_or_404, redirect
from .models import Post

def post_list(request):
    posts = Post.objects.all()
    return render(request, 'blog/post_list.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'blog/post_detail.html', {'post': post})

def post_create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        Post.objects.create(title=title, content=content)
        return redirect('post_list')
    return render(request, 'blog/post_form.html')

def post_update(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == 'POST':
        post.title = request.POST.get('title')
        post.content = request.POST.get('content')
        post.save()
        return redirect('post_detail', pk=post.pk)
    return render(request, 'blog/post_form.html', {'post': post})

def post_delete(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == 'POST':
        post.delete()
        return redirect('post_list')
    return render(request, 'blog/post_confirm_delete.html', {'post': post})
```

---

### **Step 5: Create URL Patterns**
In `blog/urls.py`, define URL patterns for the blog app:  
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('<int:pk>/', views.post_detail, name='post_detail'),
    path('new/', views.post_create, name='post_create'),
    path('<int:pk>/edit/', views.post_update, name='post_update'),
    path('<int:pk>/delete/', views.post_delete, name='post_delete'),
]
```

Include these URLs in the project’s `urls.py`:  
```python
from django.urls import path, include

urlpatterns = [
    ...,
    path('blog/', include('blog.urls')),
]
```

---

### **Step 6: Create Templates**
#### **`templates/blog/post_list.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Blog Posts</title>
</head>
<body>
    <h1>Blog Posts</h1>
    <a href="{% url 'post_create' %}">Create New Post</a>
    <ul>
        {% for post in posts %}
            <li>
                <a href="{% url 'post_detail' post.pk %}">{{ post.title }}</a>
                ({{ post.created_at }})
            </li>
        {% endfor %}
    </ul>
</body>
</html>
```

#### **`templates/blog/post_detail.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ post.title }}</title>
</head>
<body>
    <h1>{{ post.title }}</h1>
    <p>{{ post.content }}</p>
    <p><small>Last updated: {{ post.updated_at }}</small></p>
    <a href="{% url 'post_update' post.pk %}">Edit</a>
    <form action="{% url 'post_delete' post.pk %}" method="post" style="display:inline;">
        {% csrf_token %}
        <button type="submit">Delete</button>
    </form>
    <a href="{% url 'post_list' %}">Back to List</a>
</body>
</html>
```

#### **`templates/blog/post_form.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Create/Edit Post</title>
</head>
<body>
    <h1>{% if post %}Edit{% else %}Create{% endif %} Post</h1>
    <form method="post">
        {% csrf_token %}
        <label for="title">Title:</label>
        <input type="text" name="title" id="title" value="{{ post.title|default:'' }}">
        <br>
        <label for="content">Content:</label>
        <textarea name="content" id="content">{{ post.content|default:'' }}</textarea>
        <br>
        <button type="submit">Save</button>
    </form>
    <a href="{% url 'post_list' %}">Back to List</a>
</body>
</html>
```

#### **`templates/blog/post_confirm_delete.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Delete Post</title>
</head>
<body>
    <h1>Are you sure you want to delete "{{ post.title }}"?</h1>
    <form method="post">
        {% csrf_token %}
        <button type="submit">Yes, delete</button>
    </form>
    <a href="{% url 'post_list' %}">Cancel</a>
</body>
</html>
```

---

### **Step 7: Run the Development Server**
Run the Django server and access the blog app:  
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/blog/` to create, read, update, and delete blog posts.

---

### **Optional Enhancements**
- Add pagination to the post list.
- Use Django forms for handling user inputs in `post_create` and `post_update` views.
- Add user authentication to restrict certain actions to logged-in users.

Would you like help with any of these enhancements?