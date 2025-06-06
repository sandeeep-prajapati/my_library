To add a MySQL database to a .NET application, you’ll need to follow these steps:

### 1. Install the MySQL Data Connector for .NET

To connect a .NET application to MySQL, you’ll need the MySQL Data Connector, `MySql.Data.EntityFrameworkCore`. 

Run the following command in your .NET project directory:

```bash
dotnet add package MySql.EntityFrameworkCore --version 8.0.33
```

This will add the MySQL Entity Framework Core package, enabling integration with MySQL databases.

### 2. Configure the Database Context in .NET

Create a `DbContext` class if you don’t already have one. This class is used by Entity Framework to interact with the database.

For example, create a file called `AppDbContext.cs`:

```csharp
using Microsoft.EntityFrameworkCore;
using MyApiApp.Models;

namespace MyApiApp
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<Product> Products { get; set; } // Replace with your models

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseMySql("YourConnectionStringHere",
                new MySqlServerVersion(new Version(8, 0, 21))); // Use your MySQL version
        }
    }
}
```

In the above code, replace `YourConnectionStringHere` with the actual connection string for your MySQL database.

### 3. Set Up Your Connection String

In your `appsettings.json` file, add a connection string for your MySQL database:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "server=localhost;port=3306;database=YourDatabaseName;user=YourUsername;password=YourPassword;"
  }
}
```

### 4. Configure the Database Context in `Program.cs`

In your `Program.cs`, set up the `DbContext` using dependency injection:

```csharp
var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseMySql(builder.Configuration.GetConnectionString("DefaultConnection"),
        new MySqlServerVersion(new Version(8, 0, 21))));

var app = builder.Build();
app.UseHttpsRedirection();
app.MapControllers();
app.Run();
```

### 5. Create Migrations and Update the Database

1. Generate a migration:
   ```bash
   dotnet ef migrations add InitialCreate
   ```
2. Apply the migration to the database:
   ```bash
   dotnet ef database update
   ```

This will create the necessary tables in your MySQL database.

### 6. Use the Database in Your Application

With the `DbContext` configured, you can now use dependency injection to access the database in your controllers:

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
        private readonly AppDbContext _context;

        public ProductsController(AppDbContext context)
        {
            _context = context;
        }

        // GET: api/products
        [HttpGet]
        public ActionResult<IEnumerable<Product>> Get()
        {
            return _context.Products.ToList();
        }
    }
}
```

### Summary

This setup will allow your .NET application to interact with a MySQL database using Entity Framework Core, handling common database operations like `Create`, `Read`, `Update`, and `Delete`.