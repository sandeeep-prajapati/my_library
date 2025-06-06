### **Binary Search for a Sorted Array**

---

### **Understanding the Problem**
Binary search is a method for efficiently finding a target value within a sorted array. It works by repeatedly dividing the search interval in half.

---

### **Trick**
Use either **iteration** or **recursion** to:
1. Compare the middle element of the current search range with the target.
2. Narrow the search range to either the left or the right half, depending on the comparison.

---

### **Methodology**

#### **Algorithm**
1. **Initialize Pointers**:
   - `low` to 0 (start of the array).
   - `high` to \(n-1\) (end of the array).

2. **Iterate or Recur**:
   - Compute the middle index: \( mid = \lfloor \frac{{low + high}}{2} \rfloor \).
   - Compare `arr[mid]` with the target:
     - If `arr[mid] == target`, return `mid`.
     - If `arr[mid] > target`, search the left half: \( high = mid - 1 \).
     - If `arr[mid] < target`, search the right half: \( low = mid + 1 \).

3. **End Search**:
   - If `low > high`, the target is not in the array.

---

### **Complexity**
- **Time**: \(O(\log n)\), as the search space is halved with each iteration.
- **Space**:
  - \(O(1)\) for iterative implementation.
  - \(O(\log n)\) for recursive implementation due to the call stack.

---

### **Python Code**

#### **Iterative Implementation**
```python
def binary_search_iterative(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2  # Calculate the middle index
        
        if arr[mid] == target:  # Target found
            return mid
        elif arr[mid] < target:  # Search the right half
            low = mid + 1
        else:  # Search the left half
            high = mid - 1

    return -1  # Target not found
```

#### **Recursive Implementation**
```python
def binary_search_recursive(arr, target, low, high):
    if low > high:  # Base case: Target not found
        return -1
    
    mid = (low + high) // 2  # Calculate the middle index
    
    if arr[mid] == target:  # Target found
        return mid
    elif arr[mid] < target:  # Search the right half
        return binary_search_recursive(arr, target, mid + 1, high)
    else:  # Search the left half
        return binary_search_recursive(arr, target, low, mid - 1)
```

---

### **Example Walkthrough**

#### **Input**:
`arr = [1, 3, 5, 7, 9, 11]`, `target = 7`

#### **Execution (Iterative)**:
1. Initialize:
   - `low = 0`, `high = 5`

2. First Iteration:
   - `mid = (0 + 5) // 2 = 2`
   - `arr[mid] = 5`
   - Since \( 5 < 7 \), search the right half: \( low = 3 \).

3. Second Iteration:
   - `mid = (3 + 5) // 2 = 4`
   - `arr[mid] = 9`
   - Since \( 9 > 7 \), search the left half: \( high = 3 \).

4. Third Iteration:
   - `mid = (3 + 3) // 2 = 3`
   - `arr[mid] = 7`
   - Target found at index 3.

#### **Output**:
`3`

---

### **Edge Cases**
1. **Target Not in Array**:
   - Input: `arr = [1, 2, 3]`, `target = 4`
   - Output: `-1`.
2. **Single Element Array**:
   - Input: `arr = [5]`, `target = 5`
   - Output: `0`.
3. **Empty Array**:
   - Input: `arr = []`, `target = 10`
   - Output: `-1`.

---

### **Tips for Binary Search**
1. **Always Check for Sorted Input**:
   - Binary search only works on sorted arrays.
2. **Middle Calculation**:
   - Avoid overflow in some languages with \( mid = low + \frac{{high - low}}{2} \).
3. **Debugging**:
   - Carefully track `low`, `high`, and `mid` during iterations to debug off-by-one errors.

---

### **Practice Variations**
1. **Find the First or Last Occurrence**:
   - Use binary search to locate the first or last occurrence of a target in a sorted array.
2. **Search in a Rotated Sorted Array**:
   - Modify binary search to handle arrays that are rotated at some pivot.
3. **Find Square Root**:
   - Use binary search to find the integer square root of a number.

Let me know if you'd like to explore any of these extensions!