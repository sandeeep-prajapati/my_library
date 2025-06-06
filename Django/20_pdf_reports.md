Creating and serving PDF reports dynamically in Django is a common use case for generating invoices, reports, or certificates. You can use libraries like `WeasyPrint` or `ReportLab` for generating PDFs. Below, I'll show you how to integrate both libraries in Django to create and serve PDF reports.

### **Option 1: Create and Serve PDF Reports Using WeasyPrint**

WeasyPrint is a modern library for rendering HTML to PDF using WebKit. It's highly recommended for generating PDFs from HTML templates, as it allows you to take full advantage of CSS for styling.

#### **Step 1: Install WeasyPrint**

1. Install the WeasyPrint library:

   ```bash
   pip install weasyprint
   ```

2. WeasyPrint requires some system dependencies, like Cairo and Pango. You may need to install them depending on your OS:

   - **On Ubuntu:**

     ```bash
     sudo apt-get install libcairo2 libcairo2-dev libpango1.0-0 libpango1.0-dev
     ```

   - **On macOS:**

     ```bash
     brew install cairo pango
     ```

#### **Step 2: Create a View to Generate PDF**

1. **In your `views.py`, import WeasyPrint and create a view to generate the PDF**:

   ```python
   from django.shortcuts import render
   from django.http import HttpResponse
   from weasyprint import HTML

   def generate_pdf(request):
       # Example data for the PDF
       context = {
           'title': 'Monthly Report',
           'content': 'This is a sample report generated with WeasyPrint.'
       }

       # Render the HTML template
       html_content = render(request, 'report_template.html', context)

       # Convert HTML to PDF using WeasyPrint
       pdf = HTML(string=html_content.content.decode('utf-8')).write_pdf()

       # Create the HTTP response and serve the PDF file
       response = HttpResponse(pdf, content_type='application/pdf')
       response['Content-Disposition'] = 'inline; filename="report.pdf"'

       return response
   ```

2. **Create an HTML template (`report_template.html`) that will be used to generate the PDF**:

   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <style>
           h1 {
               color: #2d3e50;
               text-align: center;
           }
           .content {
               font-size: 14px;
               margin-top: 20px;
           }
       </style>
   </head>
   <body>
       <h1>{{ title }}</h1>
       <div class="content">
           <p>{{ content }}</p>
       </div>
   </body>
   </html>
   ```

3. **Add a URL pattern to map the PDF generation view**:

   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('generate-pdf/', views.generate_pdf, name='generate_pdf'),
   ]
   ```

#### **Step 3: Test the PDF Generation**

- Now, when you visit the URL `/generate-pdf/`, it will generate and serve the PDF dynamically based on the HTML template and context passed to it.
- The generated PDF will be shown inline in the browser, or you can adjust the `Content-Disposition` header to trigger a download instead.

---

### **Option 2: Create and Serve PDF Reports Using ReportLab**

ReportLab is another popular library for generating PDFs in Python, allowing for more granular control over the PDF layout and content.

#### **Step 1: Install ReportLab**

1. Install ReportLab:

   ```bash
   pip install reportlab
   ```

#### **Step 2: Create a View to Generate PDF**

1. **In your `views.py`, use ReportLab to generate a PDF:**

   ```python
   from django.http import HttpResponse
   from reportlab.lib.pagesizes import letter
   from reportlab.pdfgen import canvas

   def generate_pdf(request):
       # Create a response object to serve the PDF
       response = HttpResponse(content_type='application/pdf')
       response['Content-Disposition'] = 'inline; filename="report.pdf"'

       # Create a PDF using ReportLab
       p = canvas.Canvas(response, pagesize=letter)
       p.setFont("Helvetica", 12)

       # Add content to the PDF
       p.drawString(100, 750, "Monthly Report")
       p.drawString(100, 730, "This is a sample report generated with ReportLab.")

       # Save the PDF
       p.showPage()
       p.save()

       return response
   ```

2. **Create a URL pattern for the view**:

   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('generate-pdf/', views.generate_pdf, name='generate_pdf'),
   ]
   ```

#### **Step 3: Test the PDF Generation**

- When you visit `/generate-pdf/`, it will generate a PDF using ReportLab and serve it directly to the browser.

---

### **Option 3: Serve PDF Reports with Django and AJAX**

You can also use AJAX to fetch the PDF dynamically without reloading the page, which is common in interactive web applications.

1. **Use JavaScript and AJAX to call the Django view that generates the PDF**:

   ```javascript
   function downloadPDF() {
       fetch('/generate-pdf/')
           .then(response => response.blob())
           .then(blob => {
               const link = document.createElement('a');
               link.href = URL.createObjectURL(blob);
               link.download = 'report.pdf';
               link.click();
           });
   }
   ```

2. **Add a button to trigger the AJAX call in your HTML template**:

   ```html
   <button onclick="downloadPDF()">Download PDF</button>
   ```

This allows users to download the PDF by clicking the button without leaving the page.

---

### **Conclusion**

You now have two ways to generate and serve PDF reports dynamically in Django:

1. **WeasyPrint**: Best suited for generating PDFs from HTML and CSS. It's easier if you need to generate PDFs that are heavily styled or based on complex HTML templates.
2. **ReportLab**: Provides more control over the layout and elements of the PDF, allowing you to create highly customized PDFs from scratch.

Choose the method that best fits your project’s needs!