### **Checking Valid Parentheses**

---

### **Prompt**  
**Goal**: Determine if a given string with parentheses is valid.  
A string is valid if:
1. Open brackets are closed by the same type of brackets.  
2. Open brackets are closed in the correct order.  

---

### **Trick**  
- Use a **stack** to manage open brackets and ensure they match with closing brackets in the correct order.  

---

### **Methodology**  

1. **Initialize a Stack**:  
   - Use a list in Python to act as a stack.  

2. **Define Matching Pairs**:  
   - Create a mapping of closing brackets to their corresponding open brackets for quick lookup.  

3. **Iterate Through the String**:  
   - If the character is an open bracket (`(`, `[`, `{`), push it onto the stack.  
   - If the character is a closing bracket (`)`, `]`, `}`):  
     - Check if the stack is empty (unmatched closing bracket).  
     - Compare the top of the stack with the corresponding open bracket.  
     - Pop the stack if it matches; otherwise, the string is invalid.  

4. **Final Validation**:  
   - At the end of the iteration, the stack should be empty.  
   - If not, it means there are unmatched open brackets.  

---

### **Python Implementation**  
```python
def is_valid_parentheses(s):
    # Dictionary to map closing to opening brackets
    bracket_map = {')': '(', '}': '{', ']': '['}
    stack = []

    for char in s:
        if char in bracket_map.values():  # If it's an open bracket
            stack.append(char)
        elif char in bracket_map.keys():  # If it's a closing bracket
            if not stack or stack[-1] != bracket_map[char]:  # Mismatch
                return False
            stack.pop()  # Pop the matching open bracket
        else:
            # Ignore non-bracket characters (optional for extended cases)
            continue

    return not stack  # Return True if stack is empty, False otherwise

# Example usage
print(is_valid_parentheses("()[]{}"))  # Output: True
print(is_valid_parentheses("(]"))      # Output: False
print(is_valid_parentheses("([)]"))    # Output: False
print(is_valid_parentheses("{[]}"))    # Output: True
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(n) \), where \( n \) is the length of the string. Each character is processed once.  

2. **Space Complexity**:  
   - \( O(n) \), in the worst case, all open brackets are pushed onto the stack.  

3. **Edge Cases**:  
   - Empty string: Valid (return `True`).  
   - Unbalanced brackets: Return `False`.  
   - Non-bracket characters (optional): Either ignore or consider invalid based on problem constraints.  

---

### **Example Walkthrough**  

**Input**: `"([{}])"`  

1. **Iteration 1**: `(` → Stack: `['(']`.  
2. **Iteration 2**: `[` → Stack: `['(', '[']`.  
3. **Iteration 3**: `{` → Stack: `['(', '[', '{']`.  
4. **Iteration 4**: `}` → Matches `{`. Pop stack → Stack: `['(', '[']`.  
5. **Iteration 5**: `]` → Matches `[`. Pop stack → Stack: `['(']`.  
6. **Iteration 6**: `)` → Matches `(`. Pop stack → Stack: `[]`.  

**Final Validation**: Stack is empty → Return `True`.