### **Integrate Google and Facebook OAuth for Social Media Logins using `django-allauth`**

---

### **Step 1: Install Required Packages**

Install the `django-allauth` package:  
```bash
pip install django-allauth
```

---

### **Step 2: Configure Installed Apps**

Add `allauth` and related apps to your `INSTALLED_APPS` in `settings.py`:  
```python
INSTALLED_APPS = [
    ...,
    'django.contrib.sites',  # Required by django-allauth
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'allauth.socialaccount.providers.facebook',
]
```

---

### **Step 3: Set Up Django Sites Framework**

Run migrations to enable the sites framework:  
```bash
python manage.py migrate
```

Create a site using the Django admin interface:
1. Navigate to `/admin/sites/site/`.
2. Update the `Domain Name` and `Display Name` to match your project (e.g., `localhost:8000` for development).

Alternatively, use the shell:  
```bash
from django.contrib.sites.models import Site
Site.objects.update_or_create(id=1, defaults={'domain': 'localhost:8000', 'name': 'MyProject'})
```

---

### **Step 4: Configure Allauth in `settings.py`**

Add these configurations to `settings.py`:  
```python
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',  # Default backend
    'allauth.account.auth_backends.AuthenticationBackend',  # Required by django-allauth
]

SITE_ID = 1  # Update this with your site ID
LOGIN_REDIRECT_URL = '/'  # Redirect after successful login
LOGOUT_REDIRECT_URL = '/'  # Redirect after logout
ACCOUNT_LOGOUT_ON_GET = True  # Optional: Log out immediately on GET request
```

---

### **Step 5: Set Up URLs**

Include the `allauth` URLs in your project’s `urls.py`:  
```python
from django.urls import path, include

urlpatterns = [
    path('accounts/', include('allauth.urls')),  # Django-allauth URLs
]
```

---

### **Step 6: Register OAuth Providers**

1. Log in to your Django admin panel at `/admin/`.
2. Go to `Social Applications` under `Social Accounts`.
3. Add configurations for Google and Facebook:

#### **Google Configuration**
1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable the "Google+ API" or "OAuth2.0 APIs".
3. Generate OAuth credentials (Client ID and Client Secret).
4. Add the following authorized redirect URI:  
   ```
   http://localhost:8000/accounts/google/login/callback/
   ```
5. Enter the Client ID and Client Secret in the Django admin:
   - Provider: **Google**
   - Sites: Select your site (e.g., `localhost:8000`).

#### **Facebook Configuration**
1. Visit the [Facebook Developer Console](https://developers.facebook.com/).
2. Create a new app and set up Facebook Login.
3. Generate App ID and App Secret.
4. Add the following redirect URI:  
   ```
   http://localhost:8000/accounts/facebook/login/callback/
   ```
5. Enter the App ID and App Secret in the Django admin:
   - Provider: **Facebook**
   - Sites: Select your site (e.g., `localhost:8000`).

---

### **Step 7: Test the Social Logins**

1. Run the Django development server:  
   ```bash
   python manage.py runserver
   ```
2. Visit `/accounts/login/` and test the Google and Facebook login buttons.

---

### **Optional Enhancements**
1. **Customize Signup Flow:**
   Override `account/signup.html` to control how new users are created during the login process.

2. **Add Social Login Buttons:**
   Use a template like this to list social login providers in your `account/login.html`:  
   ```html
   <h2>Log in with:</h2>
   <ul>
       {% providers_media_js %}
       {% for provider in socialaccount_providers %}
           <li><a href="{% provider_login_url provider.id %}">{{ provider.name }}</a></li>
       {% endfor %}
   </ul>
   ```

3. **Secure API Keys:**
   Use environment variables or Django’s `settings.py` to store sensitive OAuth credentials.

