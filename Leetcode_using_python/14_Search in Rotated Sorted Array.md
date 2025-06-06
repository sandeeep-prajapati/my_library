### **Search in a Rotated Sorted Array**

---

### **Prompt**  
**Goal**: Find a target value in a rotated sorted array. Return its index or `-1` if not found.  
**Example**:  
Input: `nums = [4, 5, 6, 7, 0, 1, 2]`, `target = 0`  
Output: `4`  

---

### **Trick**  
Modify the **binary search** algorithm to account for the rotation:  
1. Determine which half of the array is sorted (left or right).  
2. Adjust the search range based on the sorted portion and the target's position.  

---

### **Methodology**  

1. **Binary Search Steps**:
   - Start with `left = 0` and `right = len(nums) - 1`.  
   - Find the middle index: `mid = (left + right) // 2`.  

2. **Check If Found**:
   - If `nums[mid] == target`, return `mid`.  

3. **Determine Sorted Half**:
   - If `nums[left] <= nums[mid]`, the **left half** is sorted:
     - Check if the target lies within `nums[left]` and `nums[mid]` to decide whether to search in the left or right half.  
   - Otherwise, the **right half** is sorted:
     - Check if the target lies within `nums[mid]` and `nums[right]` to decide the search range.  

4. **Repeat**:
   - Adjust `left` or `right` based on the above checks and repeat until `left > right`.  

---

### **Python Implementation**  

```python
def search_rotated_array(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        # Check if target is found
        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Search in the left half
            else:
                left = mid + 1  # Search in the right half
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1  # Search in the right half
            else:
                right = mid - 1  # Search in the left half

    return -1  # Target not found

# Example usage
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_rotated_array(nums, target))  # Output: 4
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(\log n) \): Binary search is performed on the array.  

2. **Space Complexity**:  
   - \( O(1) \): No extra space is used.  

3. **Edge Cases**:  
   - Empty array (`nums = []`): Return `-1`.  
   - Single-element array (`nums = [1]`, `target = 1`): Return `0`.  
   - Target not in array (`nums = [1, 2, 3]`, `target = 4`): Return `-1`.  

---

### **Example Walkthrough**  

**Input**: `nums = [4, 5, 6, 7, 0, 1, 2]`, `target = 0`  

1. **Initial State**:  
   - `left = 0`, `right = 6`, `mid = 3`  
   - `nums[mid] = 7`  

2. **Left Half Sorted**:  
   - Since `nums[left] <= nums[mid]` → `[4, 5, 6, 7]` is sorted.  
   - Target (`0`) is not in `[4, 5, 6, 7]`, so adjust `left = mid + 1`.  

3. **Next State**:  
   - `left = 4`, `right = 6`, `mid = 5`  
   - `nums[mid] = 1`  

4. **Right Half Sorted**:  
   - Since `nums[mid] < nums[right]` → `[0, 1, 2]` is sorted.  
   - Target (`0`) is in `[0, 1, 2]`, so adjust `right = mid - 1`.  

5. **Final State**:  
   - `left = 4`, `right = 4`, `mid = 4`  
   - `nums[mid] = 0`, target found!  

**Output**: `4`  

This approach ensures that the rotation is handled seamlessly within \( O(\log n) \) time.