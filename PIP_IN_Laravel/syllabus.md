
---

### 🔧 CORE FUNCTIONALITY PROMPTS

1. **Create a Laravel Artisan command to initialize a Python virtual environment**
   *Prompt: `php artisan py:init` → should create `.venv` and `requirements.txt`.*

2. **Create a Laravel Artisan command to activate the Python virtual environment and install dependencies from `requirements.txt`**
   *Prompt: `php artisan py:install`*

3. **Build a controller to accept JSON input and install the specified Python libraries**
   *JSON: `{ "packages": ["numpy", "scikit-learn"] }` → runs `pip install numpy scikit-learn`.*

4. **Create a REST endpoint to return the list of currently installed Python packages**
   *Output: `{ "installed": ["flask", "requests", "pandas"] }`*

5. **Create a Laravel command to uninstall a specific Python package**
   *Prompt: `php artisan py:remove flask`*

6. **Create a Laravel service to parse and update `requirements.txt` dynamically**
   *Prompt: Automatically add `package==version` on successful installation.*

7. **Allow JSON communication to update a specific version of a package**
   *Prompt: `{ "package": "requests", "version": "2.31.0" }`*

---

### 🔄 PACKAGE MANAGEMENT PROMPTS

8. **Create a UI/Blade frontend to manage installed Python packages**
   *Buttons for "Install", "Remove", "Upgrade", "Show Info".*

9. **Create a `py:list` Artisan command to list Python packages and their versions**
   *Output: tabular format inside CLI or JSON.*

10. **Create an endpoint to export current environment as `requirements.txt`**
    *Prompt: `php artisan py:export` → saves all `pip freeze` packages.*

11. **Build a command to import and install from an existing `requirements.txt`**
    *Prompt: `php artisan py:import path/to/req.txt`*

12. **Add command to upgrade all installed packages inside virtual env**
    *Prompt: `php artisan py:update-all`*

---

### 🧪 TESTING & DEBUGGING PROMPTS

13. **Create a command to verify Python and pip are installed properly**
    *Prompt: `php artisan py:check-env` → output versions and path.*

14. **Build logging mechanism for all Python-related operations (install, remove, activate)**
    *Stored in `storage/logs/py-package.log`*

15. **Add support for checking package conflicts before installation**
    *Use `pip check` or parse `pipdeptree`*

16. **Write a feature to validate `requirements.txt` line-by-line**
    *Output warning for unknown or deprecated packages.*

---

### 🔌 ADVANCED FEATURES (Like Vite)

17. **Auto-run a background Python server (e.g., Flask) and connect to Laravel via API**
    *Prompt: `php artisan py:serve`*

18. **Integrate hot-reloading or watcher for Python files like `vite`**
    *Auto-restart Python process if script changes detected.*

19. **Implement package caching like Vite does (via `.pycache` or local pip cache)**
    *Improve performance on repeated installs.*

20. **Expose Laravel wrapper functions to run custom Python scripts via JSON**
    *POST `{ "script": "script_name.py", "args": ["arg1", "arg2"] }` → returns script output.*

---
