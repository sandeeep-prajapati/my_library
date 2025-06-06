### Notes: Finding All Subsets of a Given Set  

#### **Prompt**  
**Goal**: Find all subsets of a given set.  

#### **Trick**  
- Use **recursion** or **bit manipulation** to efficiently generate subsets.  

---

### **Methodology**  

#### 1. **Using Recursion**  
- **Concept**: At each step, decide whether to include or exclude the current element in the subset.  
- **Steps**:  
  1. Start with an empty subset (`[]`).  
  2. For each element in the set, recursively add it to existing subsets or skip it.  
  3. Base case: If all elements are processed, add the current subset to the result.  

**Python Implementation**:  
```python
def find_subsets_recursive(nums):
    def backtrack(index, current_subset):
        # Base case: all elements are processed
        if index == len(nums):
            result.append(current_subset[:])
            return
        
        # Exclude the current element
        backtrack(index + 1, current_subset)
        
        # Include the current element
        current_subset.append(nums[index])
        backtrack(index + 1, current_subset)
        current_subset.pop()  # Backtrack
        
    result = []
    backtrack(0, [])
    return result

# Example usage
nums = [1, 2, 3]
print(find_subsets_recursive(nums))
```

---

#### 2. **Using Bit Manipulation**  
- **Concept**: Represent subsets using binary numbers where each bit represents inclusion (`1`) or exclusion (`0`).  
- **Steps**:  
  1. The total number of subsets for a set of size \( n \) is \( 2^n \).  
  2. Loop through integers from `0` to \( 2^n - 1 \).  
  3. Use the binary representation of each integer to determine which elements to include.  

**Python Implementation**:  
```python
def find_subsets_bitwise(nums):
    n = len(nums)
    result = []
    
    # Loop through all possible binary representations
    for i in range(1 << n):  # 2^n combinations
        subset = []
        for j in range(n):
            if i & (1 << j):  # Check if jth bit is set
                subset.append(nums[j])
        result.append(subset)
    
    return result

# Example usage
nums = [1, 2, 3]
print(find_subsets_bitwise(nums))
```

---

### **Key Points**  
1. **Time Complexity**:  
   - Both methods generate \( 2^n \) subsets and take \( O(2^n) \) time.  
   - Recursive method has additional space overhead due to the call stack.  

2. **Space Complexity**:  
   - Recursive: \( O(n) \) for the recursion stack.  
   - Bit manipulation: \( O(1) \) additional space.  

3. **Use Cases**:  
   - Recursive method is intuitive and easy to understand.  
   - Bit manipulation is compact and faster for implementation in competitive programming.  