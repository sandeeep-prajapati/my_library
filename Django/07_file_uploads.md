### **Allow Users to Upload Files and Save Them to the Server in Django**

---

### **Step 1: Create a New App for File Uploads**
Run the following command to create a new app:  
```bash
python manage.py startapp fileupload
```

Add the app to `INSTALLED_APPS` in `settings.py`:  
```python
INSTALLED_APPS = [
    ...,
    'fileupload',
]
```

---

### **Step 2: Create a Model for File Uploads**
In `fileupload/models.py`, define a model to store uploaded files:  
```python
from django.db import models

class UploadedFile(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

Here:
- `FileField`: Stores the uploaded file.
- `upload_to`: Specifies the directory within `MEDIA_ROOT` where files will be saved.

---

### **Step 3: Configure Media Settings**
In `settings.py`, configure the media file settings:  
```python
import os

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

---

### **Step 4: Create a Form for File Uploads**
In `fileupload/forms.py`, create a form for uploading files:  
```python
from django import forms
from .models import UploadedFile

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['title', 'file']
```

---

### **Step 5: Create Views for Handling File Uploads**
In `fileupload/views.py`, create a view to handle file uploads and display uploaded files:  
```python
from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .models import UploadedFile

def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('file_list')
    else:
        form = FileUploadForm()
    return render(request, 'fileupload/upload.html', {'form': form})

def file_list(request):
    files = UploadedFile.objects.all()
    return render(request, 'fileupload/file_list.html', {'files': files})
```

---

### **Step 6: Create Templates**
#### **`templates/fileupload/upload.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Upload File</title>
</head>
<body>
    <h1>Upload File</h1>
    <form method="post" enctype="multipart/form-data">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Upload</button>
    </form>
    <a href="{% url 'file_list' %}">View Uploaded Files</a>
</body>
</html>
```

#### **`templates/fileupload/file_list.html`**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Uploaded Files</title>
</head>
<body>
    <h1>Uploaded Files</h1>
    <ul>
        {% for file in files %}
            <li>
                <a href="{{ file.file.url }}">{{ file.title }}</a> (Uploaded on: {{ file.uploaded_at }})
            </li>
        {% endfor %}
    </ul>
    <a href="{% url 'upload_file' %}">Upload Another File</a>
</body>
</html>
```

---

### **Step 7: Set Up URLs**
In `fileupload/urls.py`, define URLs for file upload and listing:  
```python
from django.urls import path
from .views import upload_file, file_list

urlpatterns = [
    path('upload/', upload_file, name='upload_file'),
    path('files/', file_list, name='file_list'),
]
```

Include these URLs in the project’s `urls.py`:  
```python
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

urlpatterns = [
    ...,
    path('fileupload/', include('fileupload.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

### **Step 8: Run Migrations**
Create and apply migrations for the `UploadedFile` model:  
```bash
python manage.py makemigrations
python manage.py migrate
```

---

### **Step 9: Test the File Upload**
1. Run the Django server:  
   ```bash
   python manage.py runserver
   ```
2. Visit `/fileupload/upload/` to upload files.
3. Visit `/fileupload/files/` to view the uploaded files.

---

### **Optional Enhancements**
1. **Restrict File Types:** Use `FileExtensionValidator` in the model to restrict file types.
   ```python
   from django.core.validators import FileExtensionValidator

   file = models.FileField(upload_to='uploads/', validators=[FileExtensionValidator(['pdf', 'docx'])])
   ```
2. **Add File Size Limit:** Write a custom validator to limit file size.
3. **Style Forms:** Use libraries like Bootstrap for a better UI.

Would you like help adding file restrictions or enhancing the UI?