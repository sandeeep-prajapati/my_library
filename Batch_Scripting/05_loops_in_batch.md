Here’s a Batch script that uses a `for` loop to print numbers from 1 to 10:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Loop through numbers from 1 to 10
for /l %%i in (1,1,10) do (
    echo %%i
)

:: Pause to let the user see the output
pause
```

3. Save the file as `print_numbers.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `print_numbers.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `print_numbers.bat`.
2. Double-click the file to run it.
3. The script will print numbers from 1 to 10 in the Command Prompt window, one number per line.

---

### **Explanation**
- `@echo off`: Prevents the display of command execution in the output.
- `for /l %%i in (1,1,10)`: A `for` loop that:
  - Starts with `1` (the first value).
  - Increments by `1` (the second value).
  - Stops at `10` (the third value).
- `echo %%i`: Prints the current value of the loop variable `%%i`.
- `pause`: Keeps the window open until the user presses a key.

You can try running this script and let me know if you encounter any issues!