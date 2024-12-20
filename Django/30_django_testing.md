Writing unit tests for your Django models, views, and forms is essential to ensure that your application behaves as expected. Here’s how you can structure tests for these components:

### **1. Testing Models**

Django provides a built-in `TestCase` class to test models. You can test model functionality such as field validation, default values, and model methods.

#### **Example: Testing a Model**

Let's assume we have a `Book` model in our application.

```python
# models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    published_date = models.DateField()

    def __str__(self):
        return f"{self.title} by {self.author}"

    def is_published(self):
        return self.published_date <= timezone.now().date()
```

We’ll write tests for the `Book` model, including testing the `__str__` method and the `is_published` method.

#### **Unit Test for Model**

```python
# tests.py
from django.test import TestCase
from .models import Book
from django.utils import timezone

class BookModelTest(TestCase):

    def setUp(self):
        # Creating a book instance for testing
        self.book = Book.objects.create(
            title="Test Book", 
            author="Test Author", 
            published_date=timezone.now().date()
        )

    def test_book_str_method(self):
        # Test the __str__ method
        self.assertEqual(str(self.book), "Test Book by Test Author")

    def test_is_published(self):
        # Test the is_published method
        self.assertTrue(self.book.is_published())

    def test_future_publish_date(self):
        # Test if a future published date returns False
        future_book = Book.objects.create(
            title="Future Book", 
            author="Future Author", 
            published_date=timezone.now().date() + timezone.timedelta(days=1)
        )
        self.assertFalse(future_book.is_published())
```

### **2. Testing Views**

Django provides the `Client` class, which allows you to simulate requests to your views. You can test the response status, content, and redirection behavior.

#### **Example: Testing a View**

Let’s say we have a view that displays a list of books.

```python
# views.py
from django.shortcuts import render
from .models import Book

def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {'books': books})
```

#### **Unit Test for View**

```python
# tests.py
from django.test import TestCase
from django.urls import reverse
from .models import Book

class BookViewTest(TestCase):

    def setUp(self):
        # Creating a book instance for testing
        self.book = Book.objects.create(
            title="Test Book", 
            author="Test Author", 
            published_date=timezone.now().date()
        )

    def test_book_list_view(self):
        # Test the book_list view
        response = self.client.get(reverse('book_list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'book_list.html')
        self.assertContains(response, "Test Book")
        self.assertContains(response, "Test Author")
```

### **3. Testing Forms**

Django forms can be tested by checking whether they validate and save data correctly.

#### **Example: Testing a Form**

Let’s assume we have a form for creating a `Book`.

```python
# forms.py
from django import forms
from .models import Book

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title', 'author', 'published_date']
```

#### **Unit Test for Form**

```python
# tests.py
from django.test import TestCase
from .forms import BookForm
from django.utils import timezone

class BookFormTest(TestCase):

    def test_form_valid(self):
        # Test if form is valid
        form_data = {'title': 'New Book', 'author': 'New Author', 'published_date': timezone.now().date()}
        form = BookForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_form_invalid(self):
        # Test if form is invalid when missing data
        form_data = {'title': '', 'author': 'New Author', 'published_date': timezone.now().date()}
        form = BookForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)
```

### **4. Testing Model Validations**

If you have custom model validation logic (e.g., checking that the published date cannot be in the future), you can test it as well.

#### **Example: Custom Model Validation**

Let's say you have a custom model method that checks if the published date is not in the future.

```python
# models.py
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    published_date = models.DateField()

    def clean(self):
        # Custom validation to check if the published date is in the future
        if self.published_date > timezone.now().date():
            raise ValidationError("Published date cannot be in the future.")
```

#### **Unit Test for Custom Validation**

```python
# tests.py
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone
from .models import Book

class BookModelValidationTest(TestCase):

    def test_published_date_in_future(self):
        # Test if future published date raises validation error
        future_book = Book(
            title="Future Book",
            author="Future Author",
            published_date=timezone.now().date() + timezone.timedelta(days=1)
        )
        with self.assertRaises(ValidationError):
            future_book.clean()
```

### **5. Running the Tests**

To run the tests, use Django’s test runner:

```bash
python manage.py test
```

Django will automatically discover and run all the test methods (those that start with `test_`) in files named `tests.py`.

---

### **Conclusion**

Unit tests in Django ensure that your models, views, and forms behave correctly under various conditions. By testing for things like:

- Model methods and custom validations.
- Views and their responses.
- Form validations and error handling.

You can ensure that your application behaves as expected and catches bugs early in the development process.