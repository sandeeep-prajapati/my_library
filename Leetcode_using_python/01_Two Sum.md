
---

### **Understanding the Problem**
1. You are given an array of integers, `nums`, and a target sum, `target`.
2. Your task is to find the indices of two numbers in the array that add up to the target.
3. Assume there is exactly one solution, and you cannot use the same element twice.

---

### **Approach**

#### **1. Naive Approach (Brute Force)**  
Check all pairs of numbers to see if their sum equals the target.  
**Algorithm**:
- Use two nested loops:
  - Outer loop iterates through each element.
  - Inner loop checks all subsequent elements for a match.  
**Complexity**:
- **Time**: \(O(n^2)\) (due to nested loops).
- **Space**: \(O(1)\).  

---

#### **2. Optimized Approach (Using a Dictionary)**  
**Key Insight**: For each number `nums[i]`, calculate the difference `diff = target - nums[i]`.  
- If `diff` is already in the dictionary, it means you've found the two numbers.
- Otherwise, store the current number with its index in the dictionary.  

**Algorithm**:
1. Initialize an empty dictionary (`num_to_index`) to store numbers and their indices.
2. Iterate through the array:
   - Compute `diff = target - nums[i]`.
   - Check if `diff` exists in `num_to_index`:
     - If yes, return the indices `[num_to_index[diff], i]`.
     - Otherwise, store the current number and its index in `num_to_index`.

**Complexity**:
- **Time**: \(O(n)\) (single pass through the array).
- **Space**: \(O(n)\) (for the dictionary).

---

### **Python Code**

```python
def two_sum(nums, target):
    # Dictionary to store numbers and their indices
    num_to_index = {}
    
    # Iterate through the array
    for i, num in enumerate(nums):
        # Calculate the difference
        diff = target - num
        
        # Check if the difference is already in the dictionary
        if diff in num_to_index:
            return [num_to_index[diff], i]
        
        # Store the current number and its index in the dictionary
        num_to_index[num] = i
```

---

### **Example Walkthrough**

#### Input:  
`nums = [2, 7, 11, 15]`, `target = 9`  

#### Execution:
1. **Initialize `num_to_index = {}`.**
2. **First iteration (i=0, num=2):**
   - Compute `diff = 9 - 2 = 7`.
   - `7` is not in `num_to_index`.
   - Add `2: 0` to `num_to_index`.  
   - `num_to_index = {2: 0}`.
3. **Second iteration (i=1, num=7):**
   - Compute `diff = 9 - 7 = 2`.
   - `2` is in `num_to_index` with index `0`.
   - Return `[0, 1]`.

---

### **Tricks and Insights**
1. **Avoid Redundant Pairs**: The dictionary ensures you don’t revisit pairs unnecessarily.
2. **Order of Indices**: Ensure you return the indices in the correct order as per the prompt.
3. **Debugging**: Use print statements to check `num_to_index` and `diff` at each step if stuck.

---

### **Edge Cases**
1. **Single Element**: Input like `[1]`, `target = 2` should return an error or a special value since no pair exists.
2. **Duplicate Numbers**: Input like `[3, 3]`, `target = 6` should handle duplicates correctly.
3. **Negative Numbers**: Input like `[-1, -2, -3, -4]`, `target = -6` should work seamlessly.

---

### **Practice Problems**
- Extend the solution for:
  1. Multiple solutions (return all pairs).
  2. Indices or values based on the requirement.
  3. Finding three numbers that sum to a target (3Sum).
