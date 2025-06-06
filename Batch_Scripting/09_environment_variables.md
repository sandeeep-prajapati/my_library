Here’s a Batch script that displays the values of environment variables like `PATH`, `USERNAME`, and `OS`:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Display the value of the PATH environment variable
echo PATH: %PATH%

:: Display the value of the USERNAME environment variable
echo USERNAME: %USERNAME%

:: Display the value of the OS environment variable
echo OS: %OS%

:: Pause to let the user see the results
pause
```

3. Save the file as `display_env_vars.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `display_env_vars.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `display_env_vars.bat`.
2. Double-click the file to run it.
3. The script will display the values of `PATH`, `USERNAME`, and `OS` in the Command Prompt window.

---

### **Explanation**
- `echo PATH: %PATH%`: Displays the value of the `PATH` environment variable.
- `echo USERNAME: %USERNAME%`: Displays the value of the `USERNAME` environment variable.
- `echo OS: %OS%`: Displays the value of the `OS` environment variable.
- `pause`: Keeps the window open until the user presses a key, allowing them to see the output.

This script will display the environment variables and their respective values, and you can modify it to check additional variables if needed! Let me know if you need further help!