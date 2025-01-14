Here is a Batch script that lists all files and folders in the current directory and saves the output to a file named `directory_list.txt`:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: List all files and folders in the current directory
:: and save the output to directory_list.txt
dir > directory_list.txt

:: Notify the user
echo The list of files and folders has been saved to directory_list.txt.

:: Pause to let the user see the message
pause
```

3. Save the file as `list_directory.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `list_directory.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `list_directory.bat`.
2. Double-click the file to run it.
3. The script will generate a list of all files and folders in the current directory and save it to a file named `directory_list.txt` in the same directory.

---

### **Explanation**
- `@echo off`: Prevents commands from being displayed in the output.
- `dir > directory_list.txt`: Lists all files and folders in the current directory and saves the output to `directory_list.txt`.
- `echo`: Displays a message indicating that the list has been saved.
- `pause`: Keeps the window open so you can view the message before it closes.

Try running this script, and let me know if you need further assistance!