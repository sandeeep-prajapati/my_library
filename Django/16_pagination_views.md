To implement **pagination** in Django, follow these steps to display content (like a list of blog posts or items) across multiple pages.

### **Step 1: Set Up Your Django Model**

Assuming you already have a model to display data. For example, if you're creating a blog, you might have a `Post` model.

```python
# models.py
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

### **Step 2: Create a View with Pagination**

In Django, pagination can be easily added using the built-in `Paginator` class. You'll need to modify your view to paginate the results.

```python
# views.py
from django.core.paginator import Paginator
from django.shortcuts import render
from .models import Post

def post_list(request):
    posts = Post.objects.all().order_by('-created_at')  # Get all posts, ordered by creation date
    paginator = Paginator(posts, 5)  # Show 5 posts per page
    page_number = request.GET.get('page')  # Get the current page number from the URL
    page_obj = paginator.get_page(page_number)  # Get the page object for the current page

    return render(request, 'post_list.html', {'page_obj': page_obj})
```

Here, the `Paginator` class is used to divide the `Post` queryset into pages, with each page displaying a maximum of 5 posts. The `get_page()` method retrieves the correct page based on the `page` query parameter.

### **Step 3: Create a Template to Display Paginated Content**

In your template, you’ll need to loop over the items in the current page and add links to navigate to other pages.

```html
<!-- post_list.html -->
{% for post in page_obj %}
    <div class="post">
        <h2>{{ post.title }}</h2>
        <p>{{ post.content|truncatewords:30 }}</p>
        <p><em>Created at: {{ post.created_at }}</em></p>
    </div>
{% endfor %}

<div class="pagination">
    <span class="step-links">
        {% if page_obj.has_previous %}
            <a href="?page=1">&laquo; first</a>
            <a href="?page={{ page_obj.previous_page_number }}">previous</a>
        {% endif %}
        
        <span class="current">
            Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}.
        </span>

        {% if page_obj.has_next %}
            <a href="?page={{ page_obj.next_page_number }}">next</a>
            <a href="?page={{ page_obj.paginator.num_pages }}">last &raquo;</a>
        {% endif %}
    </span>
</div>
```

### **Step 4: Add URL Mapping**

Next, create a URL pattern for the `post_list` view to be accessible via the browser.

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
]
```

### **Step 5: Apply Styling (Optional)**

You can style the pagination controls in your CSS to make them more user-friendly:

```css
/* styles.css */
.pagination {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.pagination .step-links a {
    margin: 0 10px;
    text-decoration: none;
}

.pagination .step-links .current {
    font-weight: bold;
}
```

### **Step 6: Test the Pagination**

Once everything is set up:

1. Run migrations to ensure the database is up to date.
   ```bash
   python manage.py migrate
   ```

2. Create some sample data via the Django admin or shell:
   ```python
   python manage.py shell
   >>> from blog.models import Post
   >>> Post.objects.create(title="Post 1", content="This is the content of post 1.")
   >>> Post.objects.create(title="Post 2", content="This is the content of post 2.")
   >>> # Create several more posts...
   ```

3. Run your Django server:
   ```bash
   python manage.py runserver
   ```

4. Visit `http://127.0.0.1:8000/` in your browser. You should see the list of posts, with pagination controls to navigate through the pages.

### **Explanation**

- **Paginator**: Divides the `Post` queryset into pages.
- **get_page()**: Retrieves the posts for the current page.
- **page_obj**: Contains the current page's data and information, like the current page number and total number of pages.
- **Template**: Loops over the `page_obj` to display each post, with pagination controls to navigate between pages.

This setup allows you to paginate through a list of items (posts, products, etc.) and manage content across multiple pages.