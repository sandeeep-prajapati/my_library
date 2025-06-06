### **Longest Consecutive Sequence**

---

### **Prompt**  
**Goal**: Find the length of the longest sequence of consecutive integers in an array.  

---

### **Trick**  
- Use a **set** to store all numbers in the array for \( O(1) \) lookup.  
- For each number in the set, check if it is the start of a sequence (i.e., `num - 1` is not in the set).  
- If it's the start, count the length of the sequence by incrementing until `num + k` is no longer in the set.  

---

### **Methodology**  

1. **Store All Numbers in a Set**:  
   - Create a set from the array to allow quick lookups and avoid duplicates.  

2. **Identify Sequence Starts**:  
   - For each number, check if it is the start of a sequence by verifying that `num - 1` is not in the set.  

3. **Count Sequence Length**:  
   - Starting from the current number, count the consecutive numbers in the sequence (`num`, `num + 1`, ...).  

4. **Track the Maximum Length**:  
   - Update the maximum length found during the process.  

---

### **Python Implementation**  

```python
def longest_consecutive(nums):
    if not nums:
        return 0

    num_set = set(nums)  # Store numbers in a set
    max_length = 0

    for num in num_set:
        # Only start counting if `num` is the start of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            # Count the length of the sequence
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            # Update the maximum length
            max_length = max(max_length, current_length)

    return max_length

# Example usage
nums = [100, 4, 200, 1, 3, 2]
print(longest_consecutive(nums))  # Output: 4 (sequence: [1, 2, 3, 4])
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(n) \): Each number is checked at most twice (once for starting a sequence and once during the sequence count).  

2. **Space Complexity**:  
   - \( O(n) \): For storing numbers in the set.  

3. **Edge Cases**:  
   - Empty array (`nums = []`): Return `0`.  
   - Single element (`nums = [10]`): Return `1`.  
   - Already sorted sequence (`nums = [1, 2, 3, 4]`): Return the length of the array.  

---

### **Example Walkthrough**  

**Input**: `nums = [100, 4, 200, 1, 3, 2]`  

1. **Create Set**:  
   `num_set = {100, 4, 200, 1, 3, 2}`  

2. **Process Each Number**:  
   - Check `100`: `100 - 1` not in the set → Start sequence → Length = `1`.  
   - Check `4`: `4 - 1` in the set → Skip.  
   - Check `200`: `200 - 1` not in the set → Start sequence → Length = `1`.  
   - Check `1`: `1 - 1` not in the set → Start sequence → Count: `1 → 2 → 3 → 4` → Length = `4`.  
   - Check `3` and `2`: Skip as their starts are already covered.  

3. **Result**: Maximum length = `4` (sequence `[1, 2, 3, 4]`).  

**Output**: `4`  

This method ensures the most efficient and accurate result for finding the longest consecutive sequence.