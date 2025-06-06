Here's a step-by-step guide to use Django's built-in authentication system to allow users to register, log in, and log out:

---

### **Step 1: Set Up Django Authentication**
Django provides a built-in user authentication system that includes models, views, and forms. Make sure your `INSTALLED_APPS` in `settings.py` includes:  
```python
INSTALLED_APPS = [
    ...,
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

Run migrations to ensure the authentication tables are created:  
```bash
python manage.py migrate
```

---

### **Step 2: Create a Django App**
Create a new app to manage user authentication:  
```bash
python manage.py startapp accounts
```

Add the app to your `INSTALLED_APPS`:  
```python
INSTALLED_APPS = [
    ...,
    'accounts',
]
```

---

### **Step 3: User Registration**
#### **Create a Registration Form**
In `accounts/forms.py`, create a custom user registration form using Django's `UserCreationForm`:  
```python
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
```

#### **Create a Registration View**
In `accounts/views.py`, create a view to handle user registration:  
```python
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import RegistrationForm

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful. You can now log in.')
            return redirect('login')
    else:
        form = RegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})
```

#### **Create a Registration Template**
In `accounts/templates/accounts/register.html`, add a simple form:  
```html
<h2>Register</h2>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Register</button>
</form>
```

---

### **Step 4: User Login**
#### **Create a Login View**
Django provides a built-in `LoginView`. You just need to configure it in your URLs.

---

### **Step 5: User Logout**
#### **Create a Logout View**
Django also provides a built-in `LogoutView`.

---

### **Step 6**

### **Step 6: URL Configuration**
Add URLs for registration, login, and logout in `accounts/urls.py`:  
```python
from django.urls import path
from django.contrib.auth import views as auth_views
from .views import register

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
]
```

Include these URLs in the project’s `urls.py`:  
```python
from django.urls import path, include

urlpatterns = [
    path('accounts/', include('accounts.urls')),
]
```

---

### **Step 7: Create Login and Logout Templates**
#### **Login Template**
Create `accounts/templates/accounts/login.html`:  
```html
<h2>Login</h2>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="{% url 'register' %}">Register here</a>.</p>
```

#### **Logout Confirmation Template (Optional)**
You can display a confirmation after logging out by customizing the `LogoutView`.  

---

### **Step 8: Test the Authentication**
1. Run the server:  
   ```bash
   python manage.py runserver
   ```
2. Visit `/accounts/register/` to register a new user.  
3. Visit `/accounts/login/` to log in with the registered credentials.  
4. After logging in, visit `/accounts/logout/` to log out.  

---

### **Optional Enhancements**
- Redirect users to a custom page after login/logout using the `LOGIN_REDIRECT_URL` and `LOGOUT_REDIRECT_URL` settings:
  ```python
  LOGIN_REDIRECT_URL = '/'
  LOGOUT_REDIRECT_URL = '/'
  ```
- Add **password reset** and **change password** functionality using Django’s built-in views:
  - `PasswordResetView`
  - `PasswordChangeView`
- Use `messages` to display feedback like "Successfully logged in" or "Logged out successfully."  
- Style your forms and templates with a CSS framework like **Bootstrap**.

Let me know if you'd like to explore these features!