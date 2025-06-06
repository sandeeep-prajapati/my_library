### **Merge Overlapping Intervals**

---

### **Prompt**  
**Goal**: Merge all overlapping intervals in a list and return the resulting list of intervals.  
**Example**:  
Input: `intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]`  
Output: `[[1, 6], [8, 10], [15, 18]]`  

---

### **Trick**  
1. **Sort Intervals**:  
   - Sort the intervals by their starting points.  
2. **Merge Logic**:  
   - Compare the **end** of the current interval with the **start** of the next interval.  
   - If they overlap, merge them by updating the end of the current interval.  
   - Otherwise, add the current interval to the result and move to the next.  

---

### **Methodology**  

1. **Sort the Intervals**:  
   - Use `sorted(intervals, key=lambda x: x[0])` to sort based on the start.  

2. **Iterate Through Intervals**:  
   - Initialize a `merged` list to store the merged intervals.  
   - For each interval:
     - If the current interval does not overlap with the last interval in `merged`, add it directly.  
     - Otherwise, merge them by updating the end of the last interval in `merged`.  

3. **Output the Result**:  
   - Return the `merged` list.  

---

### **Python Implementation**  

```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals by starting point
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]  # Initialize with the first interval

    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:  # Overlapping intervals
            prev[1] = max(prev[1], current[1])  # Merge
        else:
            merged.append(current)  # No overlap, add to result

    return merged

# Example usage
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge_intervals(intervals))  # Output: [[1, 6], [8, 10], [15, 18]]
```

---

### **Key Points**  

1. **Time Complexity**:  
   - Sorting: \( O(n \log n) \).  
   - Merging: \( O(n) \).  
   - Total: \( O(n \log n) \).  

2. **Space Complexity**:  
   - \( O(n) \): To store the merged intervals.  

3. **Edge Cases**:  
   - Empty list (`intervals = []`): Return an empty list.  
   - Single interval (`intervals = [[1, 2]]`): Return the same interval.  
   - Fully overlapping intervals (`intervals = [[1, 4], [2, 3]]`): Return a single interval.  

---

### **Example Walkthrough**  

**Input**: `intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]`  

1. **Sort Intervals**:  
   Sorted intervals: `[[1, 3], [2, 6], [8, 10], [15, 18]]`.  

2. **Merge Intervals**:  
   - Start with `merged = [[1, 3]]`.  
   - Compare `[2, 6]` with `[1, 3]` → Overlap → Merge: `merged = [[1, 6]]`.  
   - Compare `[8, 10]` with `[1, 6]` → No overlap → Add: `merged = [[1, 6], [8, 10]]`.  
   - Compare `[15, 18]` with `[8, 10]` → No overlap → Add: `merged = [[1, 6], [8, 10], [15, 18]]`.  

**Output**: `[[1, 6], [8, 10], [15, 18]]`  

This approach ensures optimal merging of overlapping intervals.