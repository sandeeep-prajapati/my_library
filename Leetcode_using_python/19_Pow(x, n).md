### **Implement `pow(x, n)` using Recursion**

---

### **Prompt**  
**Goal**: Implement a function `pow(x, n)` that calculates \( x^n \) (x raised to the power of n) using recursion.  

---

### **Trick**  
- Use **divide-and-conquer** to break down the problem.  
- Compute the result of \( x^{n//2} \) recursively and multiply the results. This helps reduce the problem size at each step.

---

### **Methodology**  

1. **Base Case**:  
   - If \( n = 0 \), return `1` (since any number raised to the power of 0 is 1).  
   - If \( n = 1 \), return `x` (since \( x^1 = x \)).

2. **Recursive Case**:  
   - For positive even `n`, use the formula:  
     \[
     x^n = (x^{n//2}) \times (x^{n//2})
     \]
   - For odd `n`, use:  
     \[
     x^n = x \times (x^{n//2}) \times (x^{n//2})
     \]

3. **Handle Negative Exponent**:  
   - If \( n \) is negative, convert the problem to positive by computing \( x^{-n} = \frac{1}{x^n} \).

4. **Divide and Conquer**:  
   - Compute the value for \( x^{n//2} \) recursively and then square it. If \( n \) is odd, multiply by `x` once more.

---

### **Python Implementation**

```python
def myPow(x, n):
    # Base case: n = 0
    if n == 0:
        return 1
    # Base case: n = 1
    if n == 1:
        return x
    # If n is negative, convert to positive exponent
    if n < 0:
        x = 1 / x
        n = -n
    
    # Recursive step
    half = myPow(x, n // 2)
    
    # If n is even, return half * half
    if n % 2 == 0:
        return half * half
    else:  # If n is odd, return x * half * half
        return x * half * half

# Example usage
print(myPow(2, 10))  # Output: 1024
print(myPow(2, -2))  # Output: 0.25
print(myPow(3, 3))   # Output: 27
```

---

### **Key Points**  

1. **Time Complexity**:  
   - The time complexity is \( O(\log n) \), since at each recursive call, \( n \) is halved.  

2. **Space Complexity**:  
   - The space complexity is \( O(\log n) \) due to the recursion stack.  

3. **Base Case Handling**:  
   - If \( n = 0 \), return `1`.  
   - If \( n = 1 \), return `x`.  
   - Negative exponents are handled by converting them into positive by taking the reciprocal.

---

### **Example Walkthrough**

1. **Input**: `myPow(2, 10)`  
   - First, compute \( 2^5 \) recursively (since 10 is even, it can be broken into \( 2^{10} = (2^5)^2 \)).
   - Then compute \( 2^2 \) recursively and return \( 2^4 \), and finally, square to get \( 2^5 \). Multiply the results as required.

2. **Output**:  
   - `1024`.

This recursive approach reduces the problem size efficiently, making it much faster than a simple iterative approach.