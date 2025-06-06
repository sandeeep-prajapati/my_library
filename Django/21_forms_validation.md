Building forms with custom validations in Django is a common task, especially when you need to ensure that user inputs meet specific criteria. Django provides a powerful form handling system that allows you to create forms, validate user input, and provide feedback.

Below, I'll walk you through the process of creating a form with custom validations for user inputs using Django forms.

### **Step 1: Create a Django Form Class**

Django provides a `forms.Form` class that you can use to define forms and handle validations. We'll create a form with some basic fields (like `username`, `email`, and `password`) and implement custom validations.

#### **Create a Form with Custom Validations**

1. **In your app’s `forms.py`, define the form with custom validation:**

   ```python
   from django import forms
   from django.core.exceptions import ValidationError
   import re

   class CustomUserForm(forms.Form):
       username = forms.CharField(max_length=100)
       email = forms.EmailField()
       password = forms.CharField(widget=forms.PasswordInput)
       confirm_password = forms.CharField(widget=forms.PasswordInput)

       def clean_username(self):
           username = self.cleaned_data.get('username')
           # Custom validation: username must only contain alphanumeric characters
           if not username.isalnum():
               raise ValidationError("Username must only contain alphanumeric characters.")
           return username

       def clean_email(self):
           email = self.cleaned_data.get('email')
           # Custom validation: check if the email domain is "example.com"
           if not email.endswith('@example.com'):
               raise ValidationError("Email must be from the domain 'example.com'.")
           return email

       def clean_password(self):
           password = self.cleaned_data.get('password')
           # Custom validation: password must have at least 8 characters
           if len(password) < 8:
               raise ValidationError("Password must have at least 8 characters.")
           return password

       def clean_confirm_password(self):
           confirm_password = self.cleaned_data.get('confirm_password')
           password = self.cleaned_data.get('password')
           # Custom validation: password and confirm password must match
           if confirm_password != password:
               raise ValidationError("Passwords do not match.")
           return confirm_password
   ```

   In this form:

   - **`clean_username`**: Validates that the username only contains alphanumeric characters.
   - **`clean_email`**: Validates that the email must come from the `example.com` domain.
   - **`clean_password`**: Ensures the password is at least 8 characters long.
   - **`clean_confirm_password`**: Validates that the `confirm_password` matches the `password`.

### **Step 2: Create a View to Handle the Form**

In the view, you can instantiate the form and handle form submission. If the form is valid, you can process the data; otherwise, return the form with error messages.

1. **In your `views.py`, create a view to handle the form submission:**

   ```python
   from django.shortcuts import render
   from .forms import CustomUserForm

   def user_registration(request):
       if request.method == 'POST':
           form = CustomUserForm(request.POST)
           if form.is_valid():
               # Process the form data (e.g., save user to the database)
               username = form.cleaned_data['username']
               email = form.cleaned_data['email']
               # Example: save user data to the database
               # user = User.objects.create(username=username, email=email)
               return render(request, 'registration_success.html', {'username': username})
           else:
               # If the form is not valid, render the form with error messages
               return render(request, 'register.html', {'form': form})
       else:
           form = CustomUserForm()
       return render(request, 'register.html', {'form': form})
   ```

2. **In your `urls.py`, add a URL pattern for the view:**

   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('register/', views.user_registration, name='register'),
   ]
   ```

### **Step 3: Create Templates for the Form and Success Page**

1. **In `register.html`, display the form and show errors if any field is invalid:**

   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>User Registration</title>
   </head>
   <body>
       <h2>User Registration</h2>

       <form method="POST">
           {% csrf_token %}
           {{ form.as_p }}

           <button type="submit">Register</button>
       </form>

       {% if form.errors %}
           <div class="errors">
               <ul>
                   {% for field in form %}
                       {% for error in field.errors %}
                           <li>{{ error }}</li>
                       {% endfor %}
                   {% endfor %}
               </ul>
           </div>
       {% endif %}
   </body>
   </html>
   ```

   This template renders the form and lists any validation errors that occur when the form is submitted.

2. **In `registration_success.html`, display a success message when the registration is successful:**

   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Registration Success</title>
   </head>
   <body>
       <h2>Registration Successful</h2>
       <p>Welcome, {{ username }}! Your registration was successful.</p>
   </body>
   </html>
   ```

### **Step 4: Test the Form**

- Navigate to the URL `/register/` and submit the form with various inputs to see how the custom validations work.
- If the form fails any validation, Django will display error messages next to the respective fields.
- If the form is successfully submitted, you'll be redirected to the success page.

### **Conclusion**

In this example, you created a custom Django form with various fields and added custom validations for each field. The form validates user inputs such as username, email, and password, and if any validation fails, Django will display the error messages on the form. This is a simple and powerful way to handle custom validation logic in Django forms.