To create a custom **Artisan command** inside your **Laravel package**, follow these structured steps. Since you’re building a package, not a regular app, you need to register and structure the command properly **within the package context**.

---

### ✅ Step-by-step Guide to Create a Laravel Artisan Command in a Package

---

### 1. **Create the Command Class**

Inside your package directory (e.g., `packages/vendor/your-package-name/`):

```bash
php artisan make:command PythonInitCommand
```

This creates the command in the default `app/Console/Commands` path — **move it** to your package folder:

```
packages/
└── your-vendor/
    └── your-package/
        └── src/
            └── Console/
                └── Commands/
                    └── PythonInitCommand.php
```

---

### 2. **Edit the Command Class**

Here’s a basic example to create `.venv` and `requirements.txt`:

```php
<?php

namespace YourVendor\YourPackage\Console\Commands;

use Illuminate\Console\Command;
use Symfony\Component\Process\Process;

class PythonInitCommand extends Command
{
    protected $signature = 'py:init';
    protected $description = 'Initialize Python virtual environment and requirements.txt';

    public function handle()
    {
        $this->info('Creating virtual environment...');
        $process = Process::fromShellCommandline('python3 -m venv .venv');
        $process->run();

        if (!file_exists(base_path('requirements.txt'))) {
            file_put_contents(base_path('requirements.txt'), '');
            $this->info('Created requirements.txt');
        }

        $this->info('Python environment initialized!');
    }
}
```

---

### 3. **Register the Command in the Package’s Service Provider**

In your package’s main service provider file (e.g., `YourPackageServiceProvider.php`):

```php
use YourVendor\YourPackage\Console\Commands\PythonInitCommand;

public function register()
{
    $this->commands([
        PythonInitCommand::class,
    ]);
}
```

---

### 4. **Autoload the Files via Composer**

In your `composer.json` of the package:

```json
"autoload": {
    "psr-4": {
        "YourVendor\\YourPackage\\": "src/"
    }
}
```

Then run:

```bash
composer dump-autoload
```

---

### 5. **Make Sure Your Package Service Provider is Registered**

In your Laravel app's `config/app.php` (if package is not auto-discovered):

```php
'providers' => [
    // ...
    YourVendor\YourPackage\YourPackageServiceProvider::class,
],
```

Or publish it via auto-discovery in your package's `composer.json`:

```json
"extra": {
    "laravel": {
        "providers": [
            "YourVendor\\YourPackage\\YourPackageServiceProvider"
        ]
    }
}
```

---

### 6. **Test It**

Run the command in your Laravel root project:

```bash
php artisan py:init
```

You should see:

```
Creating virtual environment...
Created requirements.txt
Python environment initialized!
```

---

### ✅ Done! You now have a working custom Artisan command inside your Laravel package.

Would you like me to generate a skeleton for the full package structure with a few commands pre-wired (e.g., `py:init`, `py:install`)?
