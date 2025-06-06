### **Find the k-th Largest Element in an Array**

---

### **Prompt**  
**Goal**: Given an unsorted array, find the k-th largest element.  
**Trick**: Use a **min-heap** of size `k` to keep track of the largest elements.  
**Methodology**:  
- Push elements into the heap.
- If the heap exceeds size `k`, remove the smallest element.
- The root of the heap will be the k-th largest element.

---

### **Steps**:
1. **Min-Heap**:
   - A min-heap allows us to efficiently access the smallest element in O(1) time and insert/remove elements in O(log k) time.
   - By maintaining a heap of size `k`, the smallest element in the heap will be the k-th largest element in the array.
   
2. **Algorithm**:
   - Iterate through the array:
     - Push each element into the heap.
     - If the heap size exceeds `k`, remove the smallest element.
   - After processing all elements, the root of the heap is the k-th largest element.

---

### **Python Code Implementation**:

```python
import heapq

def findKthLargest(nums, k):
    # Min-heap to store the k largest elements
    min_heap = []
    
    # Iterate through the array
    for num in nums:
        # Push current element into the min-heap
        heapq.heappush(min_heap, num)
        
        # If the heap exceeds size k, remove the smallest element
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    # The root of the heap is the k-th largest element
    return min_heap[0]

# Example usage:
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(findKthLargest(nums, k))  # Output: 5 (the 2nd largest element)
```

---

### **Explanation of the Code**:

1. **`heapq.heappush(min_heap, num)`**: Adds the current element `num` into the min-heap.
2. **`heapq.heappop(min_heap)`**: Removes the smallest element from the heap when its size exceeds `k`.
3. **Return**: After processing all elements, the smallest element in the heap is the k-th largest element in the array.

---

### **Time Complexity**:
- **`heapq.heappush`**: O(log k) for each element inserted.
- **`heapq.heappop`**: O(log k) for each removal.
- In total, for `n` elements, the time complexity is **O(n log k)**.

### **Space Complexity**:
- The space complexity is **O(k)**, as we store up to `k` elements in the heap at any time.

---

### **Example Walkthrough**:

Given the array `nums = [3, 2, 1, 5, 6, 4]` and `k = 2`, the algorithm will:

1. **Insert 3**: Heap becomes `[3]`.
2. **Insert 2**: Heap becomes `[2, 3]` (sorted after insertion).
3. **Insert 1**: Heap exceeds size `k`, pop smallest (1), heap becomes `[2, 3]`.
4. **Insert 5**: Heap becomes `[2, 3, 5]`, pop smallest (2), heap becomes `[3, 5]`.
5. **Insert 6**: Heap becomes `[3, 5, 6]`, pop smallest (3), heap becomes `[5, 6]`.
6. **Insert 4**: Heap becomes `[4, 6, 5]`, pop smallest (4), heap becomes `[5, 6]`.

After processing all elements, the root of the heap is `5`, which is the 2nd largest element.

---

### **Edge Cases**:
1. **If k is 1**: The algorithm will return the largest element.
2. **If k is equal to the length of the array**: The algorithm will return the smallest element.
3. **Negative and large numbers**: The solution works with any integer values, including negative or large numbers.

This approach efficiently solves the problem with a time complexity of **O(n log k)** and uses a heap of size **O(k)**.