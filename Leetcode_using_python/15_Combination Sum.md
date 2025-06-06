### **Combination Sum Problem**

---

### **Prompt**  
**Goal**: Find all unique combinations of numbers from a given list that sum to a target value. Each number can be used multiple times.  

---

### **Trick**  
- Use **backtracking** to explore possible combinations.  
- Include the current number and reduce the target recursively.  
- Avoid duplicates by ensuring that each number is added in a non-decreasing order during recursion.  

---

### **Methodology**  

1. **Backtracking Framework**:
   - Start with an empty combination and explore adding numbers from the list.  
   - If the target becomes 0, add the current combination to the results.  
   - If the target becomes negative, backtrack (terminate the current path).  

2. **Steps**:
   - Iterate over the list of numbers starting from the current index to maintain order.  
   - Include the current number in the combination and reduce the target.  
   - Recurse with the updated combination and target.  
   - Remove the current number from the combination (backtrack) before moving to the next number.  

3. **Avoid Duplicates**:
   - Skip numbers that would create duplicate combinations.  

---

### **Python Implementation**  

```python
def combination_sum(candidates, target):
    def backtrack(start, target, path):
        if target == 0:
            result.append(list(path))  # Valid combination
            return
        if target < 0:
            return  # Exceeded target, stop exploration

        for i in range(start, len(candidates)):
            path.append(candidates[i])  # Choose the current number
            backtrack(i, target - candidates[i], path)  # Explore further with the same number
            path.pop()  # Backtrack by removing the last number

    result = []
    backtrack(0, target, [])
    return result

# Example usage
candidates = [2, 3, 6, 7]
target = 7
print(combination_sum(candidates, target))  # Output: [[2, 2, 3], [7]]
```

---

### **Key Points**  

1. **Time Complexity**:  
   - Worst case: \( O(2^t) \), where \( t \) is the target value.  
   - Depends on the number of valid combinations and the recursive exploration.  

2. **Space Complexity**:  
   - \( O(k) \), where \( k \) is the depth of the recursion tree.  

3. **Edge Cases**:  
   - No valid combination (`candidates = [2, 4]`, `target = 1`): Return an empty list.  
   - Single-element candidate (`candidates = [3]`, `target = 6`): Return `[[3, 3]]`.  
   - Empty candidate list (`candidates = []`): Return an empty list.  

---

### **Example Walkthrough**  

**Input**: `candidates = [2, 3, 6, 7]`, `target = 7`  

1. **Initial Call**:  
   - Start at index `0` with an empty path and target `7`.  

2. **First Path**:  
   - Add `2` → Path = `[2]`, Target = `7 - 2 = 5`.  
   - Add `2` again → Path = `[2, 2]`, Target = `5 - 2 = 3`.  
   - Add `2` again → Path = `[2, 2, 2]`, Target = `3 - 2 = 1`.  
   - Add `2` again → Exceeds target. Backtrack.  
   - Try `3` → Path = `[2, 2, 3]`, Target = `3 - 3 = 0`. Valid combination!  

3. **Second Path**:  
   - Add `3` → Path = `[3]`, Target = `7 - 3 = 4`.  
   - Add `3` again → Path = `[3, 3]`, Target = `4 - 3 = 1`.  
   - Add `3` again → Exceeds target. Backtrack.  

4. **Third Path**:  
   - Add `6` → Path = `[6]`, Target = `7 - 6 = 1`.  
   - Add `6` again → Exceeds target. Backtrack.  

5. **Fourth Path**:  
   - Add `7` → Path = `[7]`, Target = `7 - 7 = 0`. Valid combination!  

**Output**: `[[2, 2, 3], [7]]`  

This ensures all unique combinations are explored without duplicates.