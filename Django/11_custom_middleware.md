### **Creating Custom Middleware to Log Request and Response Times**

In Django, middleware is a way to process requests globally before they reach the view or after the view has processed the response. You can create custom middleware to log the time taken for each request to be processed.

Here’s a step-by-step guide to create custom middleware to log request and response times.

---

### **Step 1: Create the Custom Middleware**

1. **Create a new Python file** for the middleware (e.g., `middleware.py`) inside your Django app directory.

2. **Define the middleware class** that logs the request and response times.

```python
import time
import logging

# Create a logger
logger = logging.getLogger(__name__)

class RequestResponseTimeMiddleware:
    """
    Middleware to log request and response times.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Record the start time before processing the request
        start_time = time.time()

        # Process the request and get the response
        response = self.get_response(request)

        # Calculate the time taken for the request to be processed
        end_time = time.time()
        response_time = end_time - start_time

        # Log the request method, path, and time taken
        logger.info(f"Request: {request.method} {request.path} | Response Time: {response_time:.4f} seconds")

        return response
```

### **Explanation:**
- The middleware takes the `get_response` callable as an argument, which is passed by Django to handle the response.
- The `__call__` method is invoked for every request. The start time is recorded before the request is processed, and the end time is calculated after the response is returned.
- We use `logger.info()` to log the request method (GET, POST, etc.), the request path (URL), and the time it took to process the request.

---

### **Step 2: Add Middleware to `settings.py`**

1. Open your **`settings.py`** file.
2. Add the custom middleware to the `MIDDLEWARE` list.

```python
MIDDLEWARE = [
    # Other middleware entries...
    'your_app.middleware.RequestResponseTimeMiddleware',  # Add this line
]
```

Make sure to replace `'your_app.middleware.RequestResponseTimeMiddleware'` with the correct path to where you created the `RequestResponseTimeMiddleware` class in your app.

---

### **Step 3: Configure Logging (Optional but Recommended)**

To ensure that the logs are properly output, you need to configure logging settings in `settings.py`. Here's an example of how to set it up to log the information to the console.

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        # Add a custom logger for our middleware
        'your_app': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

In this configuration:
- We log to the console at the `INFO` level.
- We specify a custom logger `your_app` to capture the logs from our middleware and show them in the console.

---

### **Step 4: Test the Middleware**

1. Run your Django development server.

```bash
python manage.py runserver
```

2. Make a few requests to your site, either through the browser or using tools like `curl` or Postman.

3. Check the logs in your console. You should see logs like the following:

```
INFO:your_app:Request: GET /home | Response Time: 0.0153 seconds
INFO:your_app:Request: POST /login | Response Time: 0.0321 seconds
```

This indicates that the middleware is successfully logging the request method, path, and the time it took to process each request.

---

### **Step 5: Handle Errors (Optional)**

To handle cases where errors occur during request processing, you can catch exceptions in your middleware and log them as well. Modify the `__call__` method like this:

```python
def __call__(self, request):
    start_time = time.time()

    try:
        response = self.get_response(request)
    except Exception as e:
        logger.error(f"Error processing request: {request.method} {request.path} | Error: {str(e)}")
        raise e  # Re-raise the exception after logging it

    end_time = time.time()
    response_time = end_time - start_time

    logger.info(f"Request: {request.method} {request.path} | Response Time: {response_time:.4f} seconds")
    return response
```

This will ensure that any exceptions are logged along with the request information.

---

### **Step 6: Clean Up and Optimize (Optional)**

- For production environments, you may want to log request times only for slow requests (e.g., above a certain threshold, like 1 second) to avoid cluttering logs with fast requests.
  
```python
if response_time > 1.0:
    logger.warning(f"Slow Request: {request.method} {request.path} | Response Time: {response_time:.4f} seconds")
```

- Consider using a more sophisticated logging system for production environments, such as logging to a file or using centralized logging solutions like **ELK Stack** or **Sentry**.

---

### **Conclusion**

Now you have a custom Django middleware that logs request and response times. You can further customize this middleware to log additional information or handle specific cases like slow requests or errors. This can help with performance monitoring and debugging during development and production.