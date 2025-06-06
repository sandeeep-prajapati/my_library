Here’s a Batch script that prompts the user for their name and displays a personalized greeting:

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Prompt the user for their name
set /p username=Please enter your name: 

:: Display a personalized greeting
echo Hello, %username%! Welcome to Batch scripting.

:: Pause to let the user see the message
pause
```

3. Save the file as `greeting.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `greeting.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `greeting.bat`.
2. Double-click the file to run it.
3. You will see a prompt asking for your name. Enter your name and press **Enter**.
4. The script will display a personalized greeting such as:

   ```
   Hello, [YourName]! Welcome to Batch scripting.
   Press any key to continue . . .
   ```

---

### **Explanation**
- `@echo off`: Hides command execution from the output.
- `set /p username=Please enter your name:`: Prompts the user to input their name and assigns it to the variable `username`.
- `echo Hello, %username%!`: Displays the personalized greeting using the value stored in `username`.
- `pause`: Keeps the window open until the user presses a key.

Try this script and let me know if you need any further help!