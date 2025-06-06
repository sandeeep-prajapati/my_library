Here’s a Batch script to print the current date and time using the built-in `date` and `time` commands:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Print the current date
echo Current Date: %date%

:: Print the current time
echo Current Time: %time%

:: Notify the user and pause the output
pause
```

3. Save the file as `current_date_time.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `current_date_time.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `current_date_time.bat`.
2. Double-click the file to run it.
3. The output will display the current date and time in the Command Prompt window.

---

### **Explanation**
- `@echo off`: Prevents commands from being echoed to the screen.
- `%date%`: Retrieves the current date from the system.
- `%time%`: Retrieves the current time from the system.
- `pause`: Ensures the window stays open until the user presses a key.

You can try this script and let me know how it works!