Creating RESTful APIs in **ASP.NET Core** is a straightforward process, thanks to its powerful features like routing, controllers, and built-in JSON serialization. Below is a step-by-step guide to creating a RESTful API in ASP.NET Core.

### Step 1: Setting Up Your ASP.NET Core Project

1. **Create a new ASP.NET Core Web API project**:
   - Open a terminal or command prompt.
   - Use the .NET CLI to create a new project:
     ```bash
     dotnet new webapi -n MyApiApp
     cd MyApiApp
     ```

2. **Open the project in your preferred IDE** (e.g., Visual Studio Code, Visual Studio).

### Step 2: Define Models
Models are used to define the data structure that will be transferred between the API and the client.

Create a `Product` model in the `Models` folder:

```csharp
namespace MyApiApp.Models
{
    public class Product
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public decimal Price { get; set; }
    }
}
```

### Step 3: Create a Controller

Controllers handle the HTTP requests and return responses. A controller is decorated with the `[ApiController]` attribute and routes are defined using the `[Route]` attribute.

1. Create a `ProductsController` in the `Controllers` folder.

```csharp
using Microsoft.AspNetCore.Mvc;
using MyApiApp.Models;
using System.Collections.Generic;
using System.Linq;

namespace MyApiApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ProductsController : ControllerBase
    {
        // In-memory list of products for demonstration purposes
        private static List<Product> products = new List<Product>
        {
            new Product { Id = 1, Name = "Product 1", Price = 19.99m },
            new Product { Id = 2, Name = "Product 2", Price = 29.99m },
            new Product { Id = 3, Name = "Product 3", Price = 39.99m }
        };

        // GET: api/products
        [HttpGet]
        public ActionResult<IEnumerable<Product>> Get()
        {
            return Ok(products); // Serializes the list into JSON and sends it as a response
        }

        // GET: api/products/{id}
        [HttpGet("{id}")]
        public ActionResult<Product> Get(int id)
        {
            var product = products.FirstOrDefault(p => p.Id == id);
            if (product == null)
            {
                return NotFound(); // Returns 404 if product is not found
            }

            return Ok(product); // Serializes the product into JSON and sends it as a response
        }

        // POST: api/products
        [HttpPost]
        public ActionResult<Product> Post([FromBody] Product newProduct)
        {
            if (newProduct == null)
            {
                return BadRequest(); // Returns 400 if the request body is invalid
            }

            newProduct.Id = products.Max(p => p.Id) + 1; // Generate new ID
            products.Add(newProduct);

            return CreatedAtAction(nameof(Get), new { id = newProduct.Id }, newProduct); // Returns 201 with the created product
        }

        // PUT: api/products/{id}
        [HttpPut("{id}")]
        public IActionResult Put(int id, [FromBody] Product updatedProduct)
        {
            var product = products.FirstOrDefault(p => p.Id == id);
            if (product == null)
            {
                return NotFound(); // Returns 404 if product is not found
            }

            product.Name = updatedProduct.Name;
            product.Price = updatedProduct.Price;

            return NoContent(); // Returns 204 for successful update with no content
        }

        // DELETE: api/products/{id}
        [HttpDelete("{id}")]
        public IActionResult Delete(int id)
        {
            var product = products.FirstOrDefault(p => p.Id == id);
            if (product == null)
            {
                return NotFound(); // Returns 404 if product is not found
            }

            products.Remove(product);

            return NoContent(); // Returns 204 for successful deletion with no content
        }
    }
}
```
for Program.cs
```
    var builder = WebApplication.CreateBuilder(args);

    // Add services to the container.
    builder.Services.AddControllers(); // Adds support for controllers, including API controllers

    // Optional: Adds Swagger for API documentation
    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerGen();

    var app = builder.Build();

    // Configure the HTTP request pipeline.
    if (app.Environment.IsDevelopment())
    {
        app.UseSwagger();
        app.UseSwaggerUI();
    }

    app.UseHttpsRedirection();

    app.UseRouting();

    app.UseAuthorization();

    app.MapControllers(); // Enables attribute routing for controllers, like ProductsController

    app.Run();

```

### Step 4: Routing
ASP.NET Core uses routing to map HTTP requests to controller actions. In the above controller:
- The `[Route("api/[controller]")]` attribute defines the base URL for the `ProductsController`.
- `[HttpGet]`, `[HttpPost]`, `[HttpPut]`, and `[HttpDelete]` define HTTP verbs for each method.
  - **GET**: Retrieves data.
  - **POST**: Creates new data.
  - **PUT**: Updates existing data.
  - **DELETE**: Removes data.

### Step 5: JSON Serialization
ASP.NET Core automatically handles JSON serialization and deserialization:
- **Serialization**: When returning an object or list (e.g., `Ok(products)` or `Ok(product)`), ASP.NET Core automatically converts it to a JSON response.
- **Deserialization**: When the client sends a request with JSON (e.g., in `POST` or `PUT` requests), ASP.NET Core automatically converts the JSON into the corresponding C# model (`Product` in this case).

This is handled using the built-in `System.Text.Json` library by default. You can customize it or use `Newtonsoft.Json` if needed.

### Step 6: Test Your API
1. Run your project:
   - In the terminal or IDE, run:
     ```bash
     dotnet run
     ```
   - Your API will be accessible at `https://localhost:5001/api/products` (or the URL shown in your console).

2. **Test the endpoints** using:
   - **GET**: `https://localhost:5001/api/products` – Fetch all products.
   - **GET**: `https://localhost:5001/api/products/{id}` – Fetch a product by its ID.
   - **POST**: `https://localhost:5001/api/products` – Add a new product (send JSON body).
   - **PUT**: `https://localhost:5001/api/products/{id}` – Update a product (send JSON body).
   - **DELETE**: `https://localhost:5001/api/products/{id}` – Delete a product.

You can use tools like **Postman** or **cURL** to test the API.

### Step 7: Additional Configuration (Optional)
To enable additional features such as Swagger for API documentation, you can install **Swashbuckle.AspNetCore**:

1. Install the NuGet package:
   ```bash
   dotnet add package Swashbuckle.AspNetCore
   ```

2. In `Startup.cs`, add Swagger configuration:

   ```csharp
   public void ConfigureServices(IServiceCollection services)
   {
       services.AddControllers();
       services.AddSwaggerGen();
   }

   public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
   {
       if (env.IsDevelopment())
       {
           app.UseDeveloperExceptionPage();
           app.UseSwagger();
           app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "My API V1"));
       }

       app.UseRouting();
       app.UseEndpoints(endpoints =>
       {
           endpoints.MapControllers();
       });
   }
   ```

3. Run the application and navigate to `https://localhost:5001/swagger` to view the API documentation.

### Conclusion
You now have a fully functional RESTful API in ASP.NET Core, with basic routing, controllers, and JSON serialization. ASP.NET Core makes it easy to build robust APIs, and the framework provides everything needed for efficient development, including automatic JSON handling, routing, and status code responses.