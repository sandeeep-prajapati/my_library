Here’s how to create an API endpoint to manage a simple resource, like books, using **Django REST Framework (DRF)**:

---

### **Step 1: Install Django REST Framework**
Install DRF in your environment:  
```bash
pip install djangorestframework
```

Add `rest_framework` to your `INSTALLED_APPS` in `settings.py`:  
```python
INSTALLED_APPS = [
    ...,
    'rest_framework',
]
```

---

### **Step 2: Create a Django App for the API**
```bash
python manage.py startapp api
```

Add the app to your `INSTALLED_APPS`:  
```python
INSTALLED_APPS = [
    ...,
    'api',
]
```

---

### **Step 3: Create the Book Model**
In `api/models.py`, define a `Book` model:  
```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    published_date = models.DateField()
    isbn = models.CharField(max_length=13, unique=True)

    def __str__(self):
        return self.title
```

Apply migrations:  
```bash
python manage.py makemigrations
python manage.py migrate
```

---

### **Step 4: Create a Serializer**
In `api/serializers.py`, define a serializer for the `Book` model:  
```python
from rest_framework import serializers
from .models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

---

### **Step 5: Create Views for the API**
In `api/views.py`, create views using DRF’s generic views or viewsets.  
Here’s an example with **class-based views**:  
```python
from rest_framework import generics
from .models import Book
from .serializers import BookSerializer

class BookListCreateView(generics.ListCreateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

class BookDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

---

### **Step 6: Set Up URLs**
In `api/urls.py`, define API routes:  
```python
from django.urls import path
from .views import BookListCreateView, BookDetailView

urlpatterns = [
    path('books/', BookListCreateView.as_view(), name='book-list-create'),
    path('books/<int:pk>/', BookDetailView.as_view(), name='book-detail'),
]
```

Include the app’s URLs in the project’s `urls.py`:  
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Include the API app's URLs
]
```

---

### **Step 7: Test the API**
Run the Django server:  
```bash
python manage.py runserver
```

Test the endpoints using a tool like **Postman** or **cURL**:
1. **GET /api/books/** - Retrieve all books.
2. **POST /api/books/** - Create a new book:
   ```json
   {
       "title": "The Great Gatsby",
       "author": "F. Scott Fitzgerald",
       "published_date": "1925-04-10",
       "isbn": "9780743273565"
   }
   ```
3. **GET /api/books/{id}/** - Retrieve a single book by its ID.
4. **PUT /api/books/{id}/** - Update a book's details.
5. **DELETE /api/books/{id}/** - Delete a book.

---

### **Optional Enhancements**
- Add pagination, filtering, or search functionality.
- Secure the API using authentication (e.g., token-based or JWT).
- Use a **ViewSet** and DRF’s **router** for more concise routing.

Let me know if you’d like to expand on any of these steps!