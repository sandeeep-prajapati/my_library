To set up Django as a backend and React as a frontend for a full-stack application, you need to integrate the two frameworks so that they can communicate effectively. Below is a step-by-step guide to setting up a Django-React full-stack application.

### **Step 1: Set Up Django Backend**

1. **Create a new Django project:**
   Start by creating a new Django project and a new app for the API.

   ```bash
   django-admin startproject fullstack_project
   cd fullstack_project
   python manage.py startapp api
   ```

2. **Install Django Rest Framework (DRF):**
   Install the Django Rest Framework to create APIs.

   ```bash
   pip install djangorestframework
   ```

   Add `rest_framework` to the `INSTALLED_APPS` in `settings.py`:

   ```python
   INSTALLED_APPS = [
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'rest_framework',
       'api',  # Your app
   ]
   ```

3. **Create a simple model in `api/models.py`:**

   For demonstration, create a simple model for storing data (e.g., `Item`).

   ```python
   # api/models.py
   from django.db import models

   class Item(models.Model):
       name = models.CharField(max_length=100)
       description = models.TextField()

       def __str__(self):
           return self.name
   ```

4. **Create serializers for the model in `api/serializers.py`:**

   Serialize the model to convert it into JSON format for the frontend.

   ```python
   # api/serializers.py
   from rest_framework import serializers
   from .models import Item

   class ItemSerializer(serializers.ModelSerializer):
       class Meta:
           model = Item
           fields = '__all__'
   ```

5. **Create views to handle API requests in `api/views.py`:**

   Create a simple view to list all `Item` objects.

   ```python
   # api/views.py
   from rest_framework.views import APIView
   from rest_framework.response import Response
   from rest_framework import status
   from .models import Item
   from .serializers import ItemSerializer

   class ItemList(APIView):
       def get(self, request):
           items = Item.objects.all()
           serializer = ItemSerializer(items, many=True)
           return Response(serializer.data, status=status.HTTP_200_OK)

       def post(self, request):
           serializer = ItemSerializer(data=request.data)
           if serializer.is_valid():
               serializer.save()
               return Response(serializer.data, status=status.HTTP_201_CREATED)
           return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
   ```

6. **Set up URLs for the API in `api/urls.py`:**

   Define the URL patterns to handle the API requests.

   ```python
   # api/urls.py
   from django.urls import path
   from .views import ItemList

   urlpatterns = [
       path('items/', ItemList.as_view(), name='item-list'),
   ]
   ```

7. **Include the API URLs in the main `urls.py` file:**

   Add the API URLs to the root URL configuration.

   ```python
   # fullstack_project/urls.py
   from django.contrib import admin
   from django.urls import path, include

   urlpatterns = [
       path('admin/', admin.site.urls),
       path('api/', include('api.urls')),  # Add API URLs here
   ]
   ```

8. **Run migrations to set up the database:**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

9. **Run the Django server:**

   ```bash
   python manage.py runserver
   ```

   The Django backend should now be accessible at `http://localhost:8000/api/items/`.

### **Step 2: Set Up React Frontend**

1. **Create a React application:**
   You can create a React app using `create-react-app`.

   In a separate directory (outside the Django project), create a new React app:

   ```bash
   npx create-react-app frontend
   cd frontend
   ```

2. **Install Axios for making HTTP requests:**

   Axios is a promise-based HTTP client used to make API requests from React.

   ```bash
   npm install axios
   ```

3. **Create a component to display data from the Django API:**

   Create a new component `ItemList.js` inside `src/components/` to fetch data from the Django API and display it.

   ```javascript
   // src/components/ItemList.js
   import React, { useState, useEffect } from 'react';
   import axios from 'axios';

   const ItemList = () => {
       const [items, setItems] = useState([]);

       useEffect(() => {
           axios.get('http://localhost:8000/api/items/')
               .then((response) => {
                   setItems(response.data);
               })
               .catch((error) => {
                   console.error("There was an error fetching the items!", error);
               });
       }, []);

       return (
           <div>
               <h1>Items</h1>
               <ul>
                   {items.map(item => (
                       <li key={item.id}>
                           <h3>{item.name}</h3>
                           <p>{item.description}</p>
                       </li>
                   ))}
               </ul>
           </div>
       );
   }

   export default ItemList;
   ```

4. **Modify the `App.js` file to use `ItemList`:**

   Import and use the `ItemList` component in `App.js`.

   ```javascript
   // src/App.js
   import React from 'react';
   import ItemList from './components/ItemList';

   function App() {
       return (
           <div className="App">
               <ItemList />
           </div>
       );
   }

   export default App;
   ```

5. **Start the React development server:**

   ```bash
   npm start
   ```

   The React app should now be running on `http://localhost:3000`.

### **Step 3: Enable Cross-Origin Resource Sharing (CORS)**

To allow the React app (running on a different port) to make requests to the Django backend, you need to configure CORS.

1. **Install `django-cors-headers`:**

   ```bash
   pip install django-cors-headers
   ```

2. **Add `corsheaders` to `INSTALLED_APPS` in `settings.py`:**

   ```python
   INSTALLED_APPS = [
       'corsheaders',
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'rest_framework',
       'api',
   ]
   ```

3. **Add CORS middleware to `MIDDLEWARE`:**

   ```python
   MIDDLEWARE = [
       'corsheaders.middleware.CorsMiddleware',
       'django.middleware.common.CommonMiddleware',
       'django.middleware.csrf.CsrfViewMiddleware',
       'django.contrib.sessions.middleware.SessionMiddleware',
       'django.middleware.security.SecurityMiddleware',
       'django.contrib.auth.middleware.AuthenticationMiddleware',
       'django.contrib.messages.middleware.MessageMiddleware',
       'django.middleware.locale.LocaleMiddleware',
   ]
   ```

4. **Allow requests from the React frontend:**

   Add this to the `settings.py` to allow CORS for the frontend:

   ```python
   CORS_ALLOWED_ORIGINS = [
       'http://localhost:3000',
   ]
   ```

### **Step 4: Test the Full-Stack Application**

- **Django Backend:** Open `http://localhost:8000/api/items/` to see the items.
- **React Frontend:** Open `http://localhost:3000/` to see the data displayed from the Django API.

### **Step 5: Optional Improvements**

1. **Handle Authentication:** Implement JWT or session-based authentication to secure the API.
2. **Form Handling:** Use forms in React to post new data to the Django API.
3. **Deployment:** Use Docker to containerize the app and deploy it on platforms like Heroku, AWS, or DigitalOcean.

### **Conclusion**

You now have a full-stack Django + React application where Django serves as the backend API and React as the frontend. This setup is the foundation for building more complex applications, and you can expand the functionality by adding user authentication, forms, and integrating more features on both the backend and frontend.