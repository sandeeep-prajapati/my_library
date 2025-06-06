### **Climbing Stairs Problem**

---

### **Prompt**  
**Goal**: Find the number of ways to climb a staircase with `n` steps, where you can take 1 or 2 steps at a time.  

---

### **Trick**  
- The problem is similar to the Fibonacci sequence, as the number of ways to climb to step `i` depends on the ways to climb to steps `i-1` and `i-2`.  

---

### **Methodology**  

1. **Define State**:  
   - Let `dp[i]` represent the number of ways to climb `i` steps.  

2. **Base Cases**:  
   - `dp[0] = 1` (one way to climb zero steps — do nothing).  
   - `dp[1] = 1` (only one way to climb one step).  

3. **Recursive Relation**:  
   - `dp[i] = dp[i-1] + dp[i-2]`  
     (Reach step `i` by taking a single step from `i-1` or a double step from `i-2`).  

4. **Iterative Solution (Bottom-Up)**:  
   - Use a loop to calculate `dp[i]` for all steps up to `n`.  

5. **Space Optimization**:  
   - Instead of maintaining a full DP array, use two variables to store the last two computed values (`dp[i-1]` and `dp[i-2]`).  

---

### **Python Implementation**  

#### Full DP Array (O(n) Space)
```python
def climb_stairs(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# Example usage
print(climb_stairs(5))  # Output: 8
```

#### Space-Optimized Solution (O(1) Space)
```python
def climb_stairs_optimized(n):
    if n <= 1:
        return 1
    prev2, prev1 = 1, 1  # Base cases: dp[0] = 1, dp[1] = 1

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1

# Example usage
print(climb_stairs_optimized(5))  # Output: 8
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(n) \): Single loop to compute results up to \( n \).  

2. **Space Complexity**:  
   - Full DP array: \( O(n) \).  
   - Optimized version: \( O(1) \).  

3. **Edge Cases**:  
   - \( n = 0 \): One way (do nothing).  
   - \( n = 1 \): One way (take one step).  

---

### **Example Walkthrough**  

**Input**: `n = 5`  

#### Iterative Solution (Full DP Array):  
- **Base Cases**: `dp[0] = 1`, `dp[1] = 1`.  
- **Step 2**: `dp[2] = dp[1] + dp[0] = 1 + 1 = 2`.  
- **Step 3**: `dp[3] = dp[2] + dp[1] = 2 + 1 = 3`.  
- **Step 4**: `dp[4] = dp[3] + dp[2] = 3 + 2 = 5`.  
- **Step 5**: `dp[5] = dp[4] + dp[3] = 5 + 3 = 8`.  

**Output**: `dp[5] = 8`  

#### Space-Optimized Solution:  
- **Start**: `prev2 = 1`, `prev1 = 1`.  
- **Step 2**: `current = prev1 + prev2 = 1 + 1 = 2` → Update: `prev2 = 1`, `prev1 = 2`.  
- **Step 3**: `current = prev1 + prev2 = 2 + 1 = 3` → Update: `prev2 = 2`, `prev1 = 3`.  
- **Step 4**: `current = prev1 + prev2 = 3 + 2 = 5` → Update: `prev2 = 3`, `prev1 = 5`.  
- **Step 5**: `current = prev1 + prev2 = 5 + 3 = 8` → Update: `prev2 = 5`, `prev1 = 8`.  

**Output**: `8`  

This approach ensures simplicity and efficiency.