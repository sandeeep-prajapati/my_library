Here’s a Batch script that checks if a file named `test.txt` exists in the current directory and displays a message accordingly:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Check if test.txt exists in the current directory
if exist test.txt (
    echo File Found
) else (
    echo File Not Found
)

:: Pause to let the user see the result
pause
```

3. Save the file as `check_file.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `check_file.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `check_file.bat`.
2. Double-click the file to run it.
3. The script will check if `test.txt` exists in the current directory and display one of the following messages:
   - `File Found` if the file exists.
   - `File Not Found` if the file does not exist.

---

### **Explanation**
- `if exist test.txt`: Checks if a file named `test.txt` exists in the current directory.
- `echo File Found`: Displays this message if the file exists.
- `echo File Not Found`: Displays this message if the file does not exist.
- `pause`: Keeps the window open until the user presses a key.

Try running the script and let me know if you need any more details!