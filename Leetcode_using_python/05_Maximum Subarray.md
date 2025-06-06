### **Finding the Contiguous Subarray with the Largest Sum**

---

### **Understanding the Problem**
Given an array of integers (which may include negative numbers), find the contiguous subarray (containing at least one number) that has the largest sum and return that sum.

---

### **Kadane's Algorithm**

Kadane's Algorithm is an efficient approach to solve this problem by keeping track of:
1. **Local maximum**: The maximum sum of the subarray ending at the current position.
2. **Global maximum**: The overall maximum sum found so far.

---

### **Methodology**

#### **Algorithm**
1. **Initialize Variables**:
   - `current_max` to track the local maximum ending at the current index.
   - `global_max` to track the overall maximum sum.

2. **Iterate Through the Array**:
   - For each element, decide whether to:
     - Include it in the current subarray (`current_max + nums[i]`).
     - Start a new subarray with the current element (`nums[i]`).
   - Update `current_max` with the maximum of these two options.

3. **Update Global Maximum**:
   - If `current_max` is greater than `global_max`, update `global_max`.

4. **Return the Result**:
   - At the end of the loop, `global_max` contains the largest sum of any contiguous subarray.

---

### **Complexity**
- **Time**: \(O(n)\), as we make a single pass through the array.
- **Space**: \(O(1)\), since no additional data structures are used.

---

### **Python Code**

```python
def max_subarray_sum(nums):
    # Initialize variables
    current_max = nums[0]
    global_max = nums[0]
    
    # Iterate through the array
    for i in range(1, len(nums)):
        # Update the current maximum
        current_max = max(nums[i], current_max + nums[i])
        
        # Update the global maximum
        if current_max > global_max:
            global_max = current_max
    
    return global_max
```

---

### **Example Walkthrough**

#### **Input**:
`nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]`

#### **Execution**:

1. Initialize:
   - `current_max = -2`
   - `global_max = -2`

2. Iterate through the array:
   - At index 1: \( nums[1] = 1 \)
     - \( current\_max = \max(1, -2 + 1) = 1 \)
     - \( global\_max = \max(-2, 1) = 1 \)
   - At index 2: \( nums[2] = -3 \)
     - \( current\_max = \max(-3, 1 - 3) = -2 \)
     - \( global\_max = \max(1, -2) = 1 \)
   - At index 3: \( nums[3] = 4 \)
     - \( current\_max = \max(4, -2 + 4) = 4 \)
     - \( global\_max = \max(1, 4) = 4 \)
   - At index 4: \( nums[4] = -1 \)
     - \( current\_max = \max(-1, 4 - 1) = 3 \)
     - \( global\_max = \max(4, 3) = 4 \)
   - At index 5: \( nums[5] = 2 \)
     - \( current\_max = \max(2, 3 + 2) = 5 \)
     - \( global\_max = \max(4, 5) = 5 \)
   - At index 6: \( nums[6] = 1 \)
     - \( current\_max = \max(1, 5 + 1) = 6 \)
     - \( global\_max = \max(5, 6) = 6 \)
   - At index 7: \( nums[7] = -5 \)
     - \( current\_max = \max(-5, 6 - 5) = 1 \)
     - \( global\_max = \max(6, 1) = 6 \)
   - At index 8: \( nums[8] = 4 \)
     - \( current\_max = \max(4, 1 + 4) = 5 \)
     - \( global\_max = \max(6, 5) = 6 \)

3. **Output**: `6`

#### Subarray:
The contiguous subarray with the largest sum is `[4, -1, 2, 1]`.

---

### **Edge Cases**
1. **All Negative Numbers**:
   - Input: `nums = [-3, -5, -1, -2]`
   - Output: `-1` (the largest single element).
2. **Single Element**:
   - Input: `nums = [7]`
   - Output: `7`.
3. **All Positive Numbers**:
   - Input: `nums = [1, 2, 3, 4]`
   - Output: `10` (sum of the entire array).

---

### **Tricks and Insights**
1. **Restarting a Subarray**:
   - Starting a new subarray is better if the current sum becomes negative.
2. **Single Pass**:
   - Kadane's Algorithm avoids the need for nested loops, making it efficient.
3. **Global and Local Tracking**:
   - Always update the global max based on the current state of the local max.

---

### **Practice Variations**
1. Return the actual subarray instead of just the sum.
2. Solve the problem for circular arrays (e.g., `nums = [5, -3, 5]`).

Let me know if you'd like to explore these variations!