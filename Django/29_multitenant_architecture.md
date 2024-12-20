Creating a multi-tenant system in Django, where tenants share or have isolated databases, involves several strategies. Multi-tenancy can be categorized into two main approaches:

1. **Shared Database with Tenant-Specific Data:** All tenants share the same database but have different tables or schema attributes to separate data.
2. **Isolated Databases for Each Tenant:** Each tenant has a completely separate database.

In this guide, we will focus on creating a multi-tenant system with **isolated databases for each tenant**, which offers stronger data isolation.

### **Step 1: Setup Django Project**

1. **Create a new Django project and app:**

   ```bash
   django-admin startproject multitenant_project
   cd multitenant_project
   python manage.py startapp tenants
   ```

2. **Add the app to `INSTALLED_APPS`:**

   Open `multitenant_project/settings.py` and add `'tenants'` to the `INSTALLED_APPS` list:

   ```python
   INSTALLED_APPS = [
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'tenants',  # Add your app here
   ]
   ```

### **Step 2: Install `django-tenant-schemas` (or `django-tenants`)**

To manage multi-tenancy with separate databases for each tenant, we can use libraries like `django-tenants` or `django-tenant-schemas`. Here, we will use `django-tenants` (an updated version of `django-tenant-schemas`).

1. Install `django-tenants` via pip:

   ```bash
   pip install django-tenants
   ```

2. Add `'django_tenants'` to `INSTALLED_APPS` in `settings.py`:

   ```python
   INSTALLED_APPS = [
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'tenants',  # Your app
       'django_tenants',  # Multi-tenant support
   ]
   ```

### **Step 3: Configure `django-tenants`**

1. **Set up database routing for multi-tenancy:**

   In your `multitenant_project/settings.py`, configure the `DATABASES` setting to use a shared "public" schema for common data and create isolated databases for tenants.

   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': 'public_database',  # Shared database for schema
           'USER': 'your_db_user',
           'PASSWORD': 'your_db_password',
           'HOST': 'localhost',
           'PORT': '5432',
       }
   }

   DATABASE_ROUTERS = ['django_tenants.routers.TenantSyncRouter']
   ```

2. **Set up tenant settings in `settings.py`:**

   Define the `TENANT_MODEL` (which contains the `Tenant` and `Domain` model).

   ```python
   TENANT_MODEL = "tenants.Tenant"  # This model will store tenant-specific data
   TENANT_DOMAIN_MODEL = "tenants.Domain"  # This model maps tenants to domains
   ```

### **Step 4: Create Tenant and Domain Models**

1. **Create the `Tenant` and `Domain` models:**

   Inside the `tenants/models.py`, create the `Tenant` and `Domain` models.

   ```python
   # tenants/models.py
   from django.db import models
   from django_tenants.models import TenantMixin

   class Tenant(TenantMixin):
       name = models.CharField(max_length=100)
       created_on = models.DateField(auto_now_add=True)
       # You can add more fields as required for your tenants

       def __str__(self):
           return self.name

   class Domain(models.Model):
       tenant = models.ForeignKey(Tenant, related_name='domains', on_delete=models.CASCADE)
       domain = models.CharField(max_length=253, unique=True)

       def __str__(self):
           return self.domain
   ```

   The `TenantMixin` adds the necessary functionality to the tenant model to manage schemas and databases. 

2. **Run migrations to create the models:**

   First, run the migrations for the `django-tenants` package to set up the shared schema (`public` schema).

   ```bash
   python manage.py migrate_schemas --shared
   ```

   Afterward, create the `Tenant` and `Domain` models:

   ```bash
   python manage.py makemigrations tenants
   python manage.py migrate_schemas
   ```

### **Step 5: Create a Custom Middleware for Tenant Routing**

You need to create a custom middleware that routes requests to the correct tenant's database based on the domain.

1. **Create middleware to identify tenant:**

   In `tenants/middleware.py`, create middleware to switch databases based on the domain.

   ```python
   # tenants/middleware.py
   from django_tenants.utils import tenant_context
   from django.http import Http404
   from django_tenants.models import Tenant

   class TenantMiddleware:
       def __init__(self, get_response):
           self.get_response = get_response

       def __call__(self, request):
           # Get domain name from request
           domain = request.get_host().split(':')[0]

           try:
               # Look for a tenant by domain
               tenant = Tenant.objects.get(domain=domain)
               tenant_context(tenant)  # Set tenant context
           except Tenant.DoesNotExist:
               raise Http404("Tenant not found")

           response = self.get_response(request)
           return response
   ```

2. **Add middleware to `settings.py`:**

   Include the new middleware in the `MIDDLEWARE` setting:

   ```python
   MIDDLEWARE = [
       'django.middleware.security.SecurityMiddleware',
       'django.contrib.sessions.middleware.SessionMiddleware',
       'django.middleware.common.CommonMiddleware',
       'django.middleware.csrf.CsrfViewMiddleware',
       'django.contrib.auth.middleware.AuthenticationMiddleware',
       'django.contrib.messages.middleware.MessageMiddleware',
       'django.middleware.clickjacking.XFrameOptionsMiddleware',
       'tenants.middleware.TenantMiddleware',  # Add this line
   ]
   ```

### **Step 6: Create Views for Tenants**

Create views to interact with tenant-specific data. For example, you could allow tenants to manage their own resources.

```python
# tenants/views.py
from django.shortcuts import render
from .models import Tenant

def tenant_dashboard(request):
    tenant = request.tenant  # Get the current tenant from the context
    return render(request, 'tenant_dashboard.html', {'tenant': tenant})
```

### **Step 7: Tenant-Specific URLs**

Define tenant-specific URLs. You can use the tenant's domain to dynamically route users to the appropriate views:

```python
# tenants/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.tenant_dashboard, name='tenant_dashboard'),
]
```

### **Step 8: Register Tenants in Admin**

Make the `Tenant` and `Domain` models accessible from Django’s admin panel for easy management.

```python
# tenants/admin.py
from django.contrib import admin
from .models import Tenant, Domain

admin.site.register(Tenant)
admin.site.register(Domain)
```

### **Step 9: Test Multi-Tenant System**

1. **Create tenants in the admin panel:**
   - After running migrations, you can add tenants in the admin panel. Go to `/admin` and add tenants and domain mappings.
   - Assign unique domains to each tenant, such as `tenant1.example.com` and `tenant2.example.com`.

2. **Access tenant-specific data:**
   - When you visit `tenant1.example.com`, Django should route you to the `Tenant` with that domain and use its isolated database.
   - Similarly, visiting `tenant2.example.com` will switch to another isolated database.

### **Step 10: Conclusion**

You now have a multi-tenant system in Django where each tenant has its own isolated database. This setup uses `django-tenants` for routing and managing tenants with separate databases.

### **Important Considerations**

- **Database Migrations:** Each tenant may have different migrations. You need to run migrations for each tenant individually when deploying new features or changes.
- **Tenant Creation:** You can automate tenant creation via API or use Django admin for manual creation.
- **Scalability:** Ensure your hosting provider supports multiple databases, or implement scaling mechanisms as your number of tenants grows.

