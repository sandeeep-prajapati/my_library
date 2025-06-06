To build a responsive website using Django templates and Bootstrap, follow the steps outlined below. This guide walks you through creating a simple, responsive website with a Django backend and Bootstrap for frontend styling.

### **Step 1: Set Up Django Project**

1. **Create a new Django project and app:**

   ```bash
   django-admin startproject responsive_site
   cd responsive_site
   python manage.py startapp website
   ```

2. **Install Bootstrap:**

   There are multiple ways to integrate Bootstrap, such as downloading it or linking it via a CDN. For simplicity, we'll use the CDN approach.

3. **Add your app to `INSTALLED_APPS`:**

   Open `responsive_site/settings.py` and add `'website'` to the `INSTALLED_APPS` list:

   ```python
   INSTALLED_APPS = [
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'website',  # Add your app here
   ]
   ```

### **Step 2: Set Up Bootstrap**

1. **Add Bootstrap to the base template:**

   Inside the `website` app, create a folder called `templates` if it doesn't exist already, and inside `templates`, create a folder named `website` to store the HTML files.

   Create a base template that will include the Bootstrap CDN and common HTML structure. In the `website/templates/website` directory, create a `base.html` file.

   ```html
   <!-- website/templates/website/base.html -->
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>{% block title %}Responsive Website{% endblock %}</title>
       <!-- Bootstrap 5 CDN -->
       <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEJ03XGf68Q6S5kUR8yK9hp5fXwLJfswS9K9I30t7tx2bRRR5SzpxOHiFbg5A" crossorigin="anonymous">
   </head>
   <body>
       <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
           <div class="container-fluid">
               <a class="navbar-brand" href="/">Home</a>
               <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                   <span class="navbar-toggler-icon"></span>
               </button>
               <div class="collapse navbar-collapse" id="navbarNav">
                   <ul class="navbar-nav ms-auto">
                       <li class="nav-item">
                           <a class="nav-link active" href="#">About</a>
                       </li>
                       <li class="nav-item">
                           <a class="nav-link" href="#">Contact</a>
                       </li>
                   </ul>
               </div>
           </div>
       </nav>

       <div class="container mt-4">
           {% block content %}
           <!-- Content will go here -->
           {% endblock %}
       </div>

       <!-- Bootstrap 5 JS and Popper.js -->
       <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js" integrity="sha384-oBqDVmMz4fnFO9gybC8HxT+xLpm6hTt9xLxwt+q4F3JQ1cxhVjyHk/cpHfi3oXTj" crossorigin="anonymous"></script>
       <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.min.js" integrity="sha384-pzjw8f+ua7Kw1TIq0G5FqkM2puj6oPpD8txA1z7RphzJ6h7x1K0I6zMjr30asPb6" crossorigin="anonymous"></script>
   </body>
   </html>
   ```

### **Step 3: Create Templates for Your Website**

1. **Create a homepage template:**

   Create a `home.html` file in the `website/templates/website` directory that extends the `base.html` template.

   ```html
   <!-- website/templates/website/home.html -->
   {% extends "website/base.html" %}

   {% block title %}Home - Responsive Website{% endblock %}

   {% block content %}
   <div class="row">
       <div class="col-md-6">
           <h1>Welcome to the Responsive Website</h1>
           <p>This website uses Django with Bootstrap to create a responsive design.</p>
       </div>
       <div class="col-md-6">
           <img src="https://via.placeholder.com/500x300" alt="Responsive Image" class="img-fluid">
       </div>
   </div>
   {% endblock %}
   ```

2. **Create an About page:**

   Create an `about.html` file in the same directory to display the About section.

   ```html
   <!-- website/templates/website/about.html -->
   {% extends "website/base.html" %}

   {% block title %}About - Responsive Website{% endblock %}

   {% block content %}
   <h2>About Us</h2>
   <p>This is an example of a website built using Django and Bootstrap. The layout is responsive, meaning it adapts to various screen sizes.</p>
   {% endblock %}
   ```

3. **Create a Contact page:**

   Similarly, create a `contact.html` file to display the Contact section.

   ```html
   <!-- website/templates/website/contact.html -->
   {% extends "website/base.html" %}

   {% block title %}Contact - Responsive Website{% endblock %}

   {% block content %}
   <h2>Contact Us</h2>
   <form>
       <div class="mb-3">
           <label for="name" class="form-label">Name</label>
           <input type="text" class="form-control" id="name" placeholder="Your Name">
       </div>
       <div class="mb-3">
           <label for="email" class="form-label">Email address</label>
           <input type="email" class="form-control" id="email" placeholder="Your Email">
       </div>
       <div class="mb-3">
           <label for="message" class="form-label">Message</label>
           <textarea class="form-control" id="message" rows="4" placeholder="Your Message"></textarea>
       </div>
       <button type="submit" class="btn btn-primary">Submit</button>
   </form>
   {% endblock %}
   ```

### **Step 4: Set Up Views and URL Routing**

1. **Create views for your templates:**

   In `website/views.py`, add the following views to render your templates:

   ```python
   # website/views.py
   from django.shortcuts import render

   def home(request):
       return render(request, 'website/home.html')

   def about(request):
       return render(request, 'website/about.html')

   def contact(request):
       return render(request, 'website/contact.html')
   ```

2. **Map URLs to views:**

   In `website/urls.py`, define the URLs for the homepage, About, and Contact pages:

   ```python
   # website/urls.py
   from django.urls import path
   from . import views

   urlpatterns = [
       path('', views.home, name='home'),
       path('about/', views.about, name='about'),
       path('contact/', views.contact, name='contact'),
   ]
   ```

3. **Include app URLs in the project URL configuration:**

   In `responsive_site/urls.py`, include the `website` app URLs:

   ```python
   # responsive_site/urls.py
   from django.contrib import admin
   from django.urls import path, include

   urlpatterns = [
       path('admin/', admin.site.urls),
       path('', include('website.urls')),  # Include website URLs
   ]
   ```

### **Step 5: Run the Development Server**

1. **Run the Django development server:**

   ```bash
   python manage.py runserver
   ```

2. **Open your website in the browser:**

   - Go to `http://localhost:8000/` for the homepage.
   - Go to `http://localhost:8000/about/` for the About page.
   - Go to `http://localhost:8000/contact/` for the Contact page.

### **Step 6: Make Your Website Responsive**

Bootstrap ensures your website is responsive by default. The grid system (`container`, `row`, `col-md-6`, etc.) automatically adjusts the layout depending on the screen size. You can customize and add more responsive classes to optimize your website for different devices.

### **Conclusion**

Now you have a fully responsive Django website using Django templates and Bootstrap. You can extend the functionality further by adding forms, user authentication, or any other features you need.