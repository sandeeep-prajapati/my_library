Here’s how to configure a Django project to use a PostgreSQL database instead of the default SQLite:

---

### **Step 1: Install PostgreSQL and psycopg2**
Ensure PostgreSQL is installed on your system. Then, install the `psycopg2` library, which is a PostgreSQL adapter for Python.  

```bash
# Install psycopg2
pip install psycopg2-binary
```

---

### **Step 2: Create a PostgreSQL Database**
1. Log in to the PostgreSQL console:  
   ```bash
   psql -U postgres
   ```
2. Create a new database:  
   ```sql
   CREATE DATABASE mydatabase;
   ```
3. Create a new user:  
   ```sql
   CREATE USER myuser WITH PASSWORD 'mypassword';
   ```
4. Grant the user privileges on the database:  
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE mydatabase TO myuser;
   ```
5. Exit the PostgreSQL console:  
   ```bash
   \q
   ```

---

### **Step 3: Update Django Settings**
Modify the `DATABASES` setting in your `myproject/settings.py` file to use PostgreSQL:  
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydatabase',          # Replace with your database name
        'USER': 'myuser',              # Replace with your database user
        'PASSWORD': 'mypassword',      # Replace with your user password
        'HOST': 'localhost',           # Set to your database host, e.g., an IP address
        'PORT': '5432',                # Default PostgreSQL port
    }
}
```

---

### **Step 4: Apply Migrations**
Run migrations to set up Django's default tables in your PostgreSQL database:  
```bash
python manage.py migrate
```

---

### **Step 5: Verify the Connection**
Start the Django development server:  
```bash
python manage.py runserver
```

Access your app and check that everything works as expected. You can confirm the connection by logging into your PostgreSQL database and verifying the tables created by Django:  
```bash
psql -U myuser -d mydatabase
```

---

### **Step 6: Optional - Test the Configuration**
To test the connection programmatically, you can use Django's `check` command:  
```bash
python manage.py check
```

If no errors appear, your configuration is successful.

---

### **Troubleshooting Tips**
- **Ensure PostgreSQL is running:** Start the PostgreSQL service if it's not active.
- **Firewall and network settings:** If connecting to a remote database, ensure the database server allows external connections.
- **Check environment variables:** Use environment variables for sensitive information like `USER` and `PASSWORD`. Libraries like `python-decouple` or `django-environ` can help with this.

Let me know if you need further assistance!