Here’s a Batch script that takes two numbers as input and performs basic arithmetic operations (addition, subtraction, multiplication, and division):

---

### **Batch Script Code**
1. Open a text editor, such as Notepad.
2. Copy and paste the following code:

```bat
@echo off
:: Prompt the user for the first number
set /p num1=Enter the first number: 

:: Prompt the user for the second number
set /p num2=Enter the second number: 

:: Perform addition
set /a sum=num1 + num2
echo The sum of %num1% and %num2% is: %sum%

:: Perform subtraction
set /a diff=num1 - num2
echo The difference between %num1% and %num2% is: %diff%

:: Perform multiplication
set /a product=num1 * num2
echo The product of %num1% and %num2% is: %product%

:: Perform division
:: Check if the second number is not zero to avoid division by zero
if %num2%==0 (
    echo Division by zero is not allowed.
) else (
    set /a quotient=num1 / num2
    echo The quotient of %num1% and %num2% is: %quotient%
)

:: Pause to let the user see the results
pause
```

3. Save the file as `arithmetic_operations.bat`:
   - In Notepad, click **File > Save As**.
   - Set the **Save as type** to "All Files."
   - Name the file `arithmetic_operations.bat`.

---

### **Run the Script**
1. Navigate to the location where you saved `arithmetic_operations.bat`.
2. Double-click the file to run it.
3. The script will prompt you to enter two numbers. After entering both numbers, it will display the results of addition, subtraction, multiplication, and division.

---

### **Explanation**
- `set /p num1=Enter the first number:`: Prompts the user to input the first number and stores it in the variable `num1`.
- `set /p num2=Enter the second number:`: Prompts the user to input the second number and stores it in the variable `num2`.
- `set /a sum=num1 + num2`: Performs addition and stores the result in `sum`.
- `set /a diff=num1 - num2`: Performs subtraction and stores the result in `diff`.
- `set /a product=num1 * num2`: Performs multiplication and stores the result in `product`.
- `set /a quotient=num1 / num2`: Performs division and stores the result in `quotient`. The division operation includes a check to ensure the second number is not zero, avoiding division by zero errors.
- `pause`: Keeps the window open so you can view the results.

Give it a try and let me know if you need any further help!