### **Rotate an Array to the Right by `k` Steps**

---

### **Prompt**  
**Goal**: Rotate an array to the right by `k` steps. For example:  
**Input**: `nums = [1, 2, 3, 4, 5, 6, 7]`, `k = 3`  
**Output**: `[5, 6, 7, 1, 2, 3, 4]`

---

### **Trick**  
Use the **reversal method**:  
- Reverse specific parts of the array to shift elements efficiently.

---

### **Methodology**  

1. **Adjust `k`**:  
   - Since rotating the array by its length results in the same array, use `k = k % len(nums)` to simplify.  

2. **Reverse the Entire Array**:  
   - Reverse all elements to bring the last `k` elements to the front (in reverse order).  

3. **Reverse the First `k` Elements**:  
   - Reverse the first `k` elements to restore their original order.  

4. **Reverse the Remaining Elements**:  
   - Reverse the remaining elements to restore their order.  

---

### **Python Implementation**  
```python
def rotate(nums, k):
    n = len(nums)
    k %= n  # Handle cases where k > n

    # Helper function to reverse a portion of the array
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    # Step 1: Reverse the entire array
    reverse(0, n - 1)
    # Step 2: Reverse the first k elements
    reverse(0, k - 1)
    # Step 3: Reverse the rest
    reverse(k, n - 1)

# Example usage
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
rotate(nums, k)
print(nums)  # Output: [5, 6, 7, 1, 2, 3, 4]
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(n) \): Each element is swapped at most twice.  

2. **Space Complexity**:  
   - \( O(1) \): No additional space is used; operations are in-place.  

3. **Edge Cases**:  
   - `k = 0`: No rotation, return the array as-is.  
   - `k > len(nums)`: Use `k % len(nums)` to simplify.  
   - Single-element array: No change.  

---

### **Example Walkthrough**  

**Input**: `nums = [1, 2, 3, 4, 5, 6, 7]`, `k = 3`  

1. **Reverse Entire Array**:  
   - Reverse `[1, 2, 3, 4, 5, 6, 7]` → `[7, 6, 5, 4, 3, 2, 1]`  

2. **Reverse First `k` Elements**:  
   - Reverse `[7, 6, 5]` → `[5, 6, 7]`  
   - Array becomes `[5, 6, 7, 4, 3, 2, 1]`  

3. **Reverse Remaining Elements**:  
   - Reverse `[4, 3, 2, 1]` → `[1, 2, 3, 4]`  
   - Array becomes `[5, 6, 7, 1, 2, 3, 4]`  

**Output**: `[5, 6, 7, 1, 2, 3, 4]`  

This method is efficient and avoids the need for additional arrays.