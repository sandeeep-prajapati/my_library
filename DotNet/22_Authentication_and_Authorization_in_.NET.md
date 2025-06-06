Implementing **authentication** and **authorization** in ASP.NET Core applications is crucial for securing your web applications and ensuring that users can access only the resources they are allowed to. ASP.NET Core provides built-in support for both through middleware, authentication schemes, and policies.

### Overview of Authentication and Authorization

- **Authentication** is the process of verifying the identity of a user or service. In ASP.NET Core, you can authenticate users using cookies, JWT (JSON Web Tokens), external providers (like Google, Facebook, etc.), or custom schemes.
- **Authorization** is the process of determining whether an authenticated user has permission to access a particular resource or perform an action.

In ASP.NET Core, the following concepts are key to implementing authentication and authorization:
- **Authentication middleware** to manage user sign-in and token validation.
- **Authorization middleware** to enforce access control based on roles, claims, or policies.

### Steps to Implement Authentication and Authorization in ASP.NET Core

### 1. **Set up Authentication in ASP.NET Core**

ASP.NET Core supports multiple authentication schemes. Let's go through the most common ones:

#### 1.1. **Cookie Authentication**
Cookie authentication is often used in web applications, where a user signs in, and an authentication cookie is set to maintain the user's session.

**Setup Cookie Authentication:**

In `Startup.cs`, in the `ConfigureServices` method, add the following:
```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(options =>
            {
                options.LoginPath = "/Account/Login";  // Redirect here if not authenticated
                options.LogoutPath = "/Account/Logout"; // Redirect here after logout
            });

    services.AddControllersWithViews();
}
```

In the `Configure` method, ensure to use authentication middleware:
```csharp
public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
{
    app.UseAuthentication();  // Enables authentication middleware
    app.UseAuthorization();   // Enables authorization middleware

    app.UseRouting();
    app.UseEndpoints(endpoints =>
    {
        endpoints.MapControllerRoute(
            name: "default",
            pattern: "{controller=Home}/{action=Index}/{id?}");
    });
}
```

**Sign In and Sign Out Logic:**
For handling sign-ins and sign-outs in your controller, use the `SignInManager` and `SignOutAsync` methods.

Example controller action for signing in:
```csharp
public class AccountController : Controller
{
    private readonly SignInManager<ApplicationUser> _signInManager;

    public AccountController(SignInManager<ApplicationUser> signInManager)
    {
        _signInManager = signInManager;
    }

    [HttpPost]
    public async Task<IActionResult> Login(string username, string password)
    {
        var user = await _userManager.FindByNameAsync(username);
        if (user != null && await _userManager.CheckPasswordAsync(user, password))
        {
            await _signInManager.SignInAsync(user, isPersistent: false);
            return RedirectToAction("Index", "Home");
        }
        return View();
    }

    public async Task<IActionResult> Logout()
    {
        await _signInManager.SignOutAsync();
        return RedirectToAction("Index", "Home");
    }
}
```

#### 1.2. **JWT (JSON Web Token) Authentication**
JWT is often used in API-based applications, especially for stateless authentication.

**Setup JWT Authentication:**

In `Startup.cs`, configure JWT bearer authentication in `ConfigureServices`:
```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuer = true,
                    ValidateAudience = true,
                    ValidateLifetime = true,
                    ValidIssuer = Configuration["Jwt:Issuer"],
                    ValidAudience = Configuration["Jwt:Audience"],
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(Configuration["Jwt:Key"]))
                };
            });

    services.AddControllers();
}
```

In `Configure` method:
```csharp
public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
{
    app.UseAuthentication();  // Add authentication middleware
    app.UseAuthorization();   // Add authorization middleware

    app.UseRouting();
    app.UseEndpoints(endpoints =>
    {
        endpoints.MapControllers();
    });
}
```

**Creating and Returning JWT Token:**

In your controller, you would create an action to issue a JWT token after user authentication.

```csharp
[HttpPost]
public IActionResult Login(string username, string password)
{
    var user = _userManager.FindByNameAsync(username).Result;

    if (user != null && _userManager.CheckPasswordAsync(user, password).Result)
    {
        var token = GenerateJwtToken(user);
        return Ok(new { token });
    }

    return Unauthorized();
}

private string GenerateJwtToken(ApplicationUser user)
{
    var claims = new[]
    {
        new Claim(JwtRegisteredClaimNames.Sub, user.UserName),
        new Claim(ClaimTypes.NameIdentifier, user.Id),
        new Claim(ClaimTypes.Role, "Admin")
    };

    var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(Configuration["Jwt:Key"]));
    var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
    var token = new JwtSecurityToken(
        issuer: Configuration["Jwt:Issuer"],
        audience: Configuration["Jwt:Audience"],
        claims: claims,
        expires: DateTime.Now.AddHours(1),
        signingCredentials: creds);

    return new JwtSecurityTokenHandler().WriteToken(token);
}
```

### 2. **Set up Authorization in ASP.NET Core**

Once authentication is configured, you need to ensure that users can only access certain resources if they have the proper permissions.

#### 2.1. **Role-based Authorization**
You can restrict access to certain controllers or actions based on the user's roles.

Example:
```csharp
[Authorize(Roles = "Admin")]
public class AdminController : Controller
{
    public IActionResult Index()
    {
        return View();
    }
}
```
In the example above, only users who are assigned the "Admin" role can access the `AdminController`.

#### 2.2. **Policy-based Authorization**
Policies offer a more flexible approach to defining authorization rules. You can use claims or other conditions to define custom authorization logic.

Example of configuring a policy in `ConfigureServices`:
```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthorization(options =>
    {
        options.AddPolicy("AdminOnly", policy => policy.RequireRole("Admin"));
    });

    services.AddControllersWithViews();
}
```

And applying the policy in the controller:
```csharp
[Authorize(Policy = "AdminOnly")]
public class AdminController : Controller
{
    public IActionResult Index()
    {
        return View();
    }
}
```

#### 2.3. **Claims-based Authorization**
Claims-based authorization uses the claims in the JWT or cookie to authorize users based on their attributes (like "Age", "Country", etc.).

Example of claim-based authorization:
```csharp
[Authorize(Policy = "Over18Only")]
public IActionResult ViewAdultContent()
{
    return View();
}
```

Configure the policy in `ConfigureServices`:
```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthorization(options =>
    {
        options.AddPolicy("Over18Only", policy => policy.RequireClaim("Age", "18"));
    });
}
```

### 3. **Handling Unauthorized Requests**
When a user is not authenticated or authorized to access a resource, ASP.NET Core can redirect them to a login page or return a 401 (Unauthorized) or 403 (Forbidden) HTTP status code, depending on the situation.

- **401 Unauthorized**: The user is not authenticated. They need to log in first.
- **403 Forbidden**: The user is authenticated but does not have the required permissions.

In ASP.NET Core, you can customize the responses for unauthorized requests by configuring options in `AddAuthentication` or `AddAuthorization`.

### Conclusion
ASP.NET Core provides robust support for both authentication and authorization, making it easy to secure your applications. You can use:
- **Cookie authentication** for session-based authentication.
- **JWT authentication** for stateless, API-based applications.
- **Role-based, policy-based, or claims-based authorization** to control access to specific resources.

By combining these features, you can ensure your ASP.NET Core applications are secure and that only authorized users can access sensitive resources.