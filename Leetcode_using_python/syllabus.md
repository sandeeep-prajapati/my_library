
---

### **1. Two Sum**
**Prompt**: Given an array of integers, find two numbers that add up to a target.  
**Trick**: Use a dictionary to store the difference between the target and the current number as you iterate.  
**Methodology**:
- Iterate through the array.
- Check if the current number exists in the dictionary.
- If yes, return the indices; otherwise, store the difference.

---

### **2. Reverse a Linked List**
**Prompt**: Reverse a singly linked list.  
**Trick**: Use three pointers (`prev`, `curr`, and `next`) to iteratively reverse the links.  
**Methodology**:
- Traverse the list, reversing pointers as you go.
- Stop when `curr` becomes `None`.

---

### **3. Merge Two Sorted Lists**
**Prompt**: Merge two sorted linked lists into one sorted list.  
**Trick**: Use a dummy node and a pointer to build the merged list.  
**Methodology**:
- Compare the heads of both lists.
- Append the smaller node to the merged list.
- Advance the pointer of the chosen list.

---

### **4. Longest Palindromic Substring**
**Prompt**: Find the longest palindromic substring in a given string.  
**Trick**: Use dynamic programming or expand around centers to check palindromes.  
**Methodology**:
- For DP, use a table to track valid palindromes.
- For expand-around-center, check both even and odd centers.

---

### **5. Maximum Subarray**
**Prompt**: Find the contiguous subarray with the largest sum.  
**Trick**: Use Kadane’s Algorithm.  
**Methodology**:
- Keep track of the maximum sum ending at each position.
- Update a global max when the local max exceeds it.

---

### **6. Binary Search**
**Prompt**: Implement binary search for a sorted array.  
**Trick**: Use a loop or recursion to divide the array into halves.  
**Methodology**:
- Compare the middle element to the target.
- Narrow the search to the left or right half.

---

### **7. Permutations**
**Prompt**: Generate all permutations of a list of numbers.  
**Trick**: Use backtracking to explore all arrangements.  
**Methodology**:
- Swap elements to generate permutations.
- Use recursion to iterate over possibilities.

---

### **8. Subsets**
**Prompt**: Find all subsets of a given set.  
**Trick**: Use recursion or bit manipulation.  
**Methodology**:
- Include or exclude each element recursively.
- For bit manipulation, treat the binary representation as inclusion/exclusion.

---

### **9. Coin Change**
**Prompt**: Given an amount and a list of coins, find the minimum number of coins needed to make the amount.  
**Trick**: Use a bottom-up DP table to store results for subproblems.  
**Methodology**:
- Initialize a table with `inf`.
- Iterate through coins to update the table.

---

### **10. Word Search**
**Prompt**: Find if a word exists in a 2D board following adjacent letters.  
**Trick**: Use DFS with backtracking to explore possible paths.  
**Methodology**:
- Mark cells as visited and backtrack after exploring.

---

### **11. Valid Parentheses**
**Prompt**: Check if a string with parentheses is valid.  
**Trick**: Use a stack to ensure proper closing of brackets.  
**Methodology**:
- Push open brackets onto the stack.
- For closing brackets, check if they match the top of the stack.

---

### **12. Rotate Array**
**Prompt**: Rotate an array to the right by `k` steps.  
**Trick**: Reverse parts of the array to achieve rotation.  
**Methodology**:
- Reverse the entire array, then reverse subarrays.

---

### **13. Climbing Stairs**
**Prompt**: Find the number of ways to climb a staircase with `n` steps where you can take 1 or 2 steps.  
**Trick**: Use Fibonacci-like DP.  
**Methodology**:
- Let `dp[i]` be the ways to climb `i` steps.
- Use `dp[i] = dp[i-1] + dp[i-2]`.

---

### **14. Search in Rotated Sorted Array**
**Prompt**: Search for a target in a rotated sorted array.  
**Trick**: Modify binary search to handle rotation.  
**Methodology**:
- Check which half of the array is sorted and adjust search accordingly.

---

### **15. Combination Sum**
**Prompt**: Find combinations of numbers that sum to a target.  
**Trick**: Use backtracking to explore combinations.  
**Methodology**:
- Include current number and reduce the target recursively.
- Avoid duplicates by skipping used numbers.

---

### **16. Merge Intervals**
**Prompt**: Merge overlapping intervals.  
**Trick**: Sort intervals and merge overlapping ones.  
**Methodology**:
- Compare ends and starts of adjacent intervals.

---

### **17. Longest Consecutive Sequence**
**Prompt**: Find the longest sequence of consecutive integers in an array.  
**Trick**: Use a set for O(1) lookup.  
**Methodology**:
- Check each number’s neighbors in the set.

---

### **18. Valid Sudoku**
**Prompt**: Check if a Sudoku board is valid.  
**Trick**: Use hash sets for rows, columns, and grids.  
**Methodology**:
- Ensure no duplicates in each row, column, and subgrid.

---

### **19. Pow(x, n)**
**Prompt**: Implement `pow(x, n)` using recursion.  
**Trick**: Use divide-and-conquer.  
**Methodology**:
- Compute `pow(x, n//2)` and multiply the results.

---

### **20. Serialize and Deserialize Tree**
**Prompt**: Serialize and deserialize a binary tree.  
**Trick**: Use BFS for serialization.  
**Methodology**:
- Encode nodes level by level with null markers.

---

### **21. Maximal Rectangle**
**Prompt**: Find the largest rectangle of 1s in a binary matrix.  
**Trick**: Use histogram-based DP.  
**Methodology**:
- Treat each row as a histogram.

---

### **22. Longest Repeating Character Replacement**
**Prompt**: Replace characters in a string to get the longest substring of repeated characters.  
**Trick**: Use a sliding window with a frequency map.  
**Methodology**:
- Adjust window size based on max character frequency.

---

### **23. Decode Ways**
**Prompt**: Decode a string of digits into letters.  
**Trick**: Use DP to count ways for each prefix.  
**Methodology**:
- Use constraints to decide valid encodings.

---

### **24. House Robber**
**Prompt**: Find the maximum amount that can be robbed without alerting the police.  
**Trick**: Use DP with two choices per house.  
**Methodology**:
- Use `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`.

---

### **25. Trap Rainwater**
**Prompt**: Find the total trapped rainwater in an elevation map.  
**Trick**: Use two-pointer technique to calculate trapped water.  
**Methodology**:
- Maintain max heights from left and right.

---

### **26. Product of Array Except Self**
**Prompt**: Return an array where each element is the product of all other elements.  
**Trick**: Use prefix and suffix arrays for O(n).  
**Methodology**:
- Combine prefix and suffix products.

---

### **27. Minimum Path Sum**
**Prompt**: Find the minimum sum to reach the bottom-right of a grid.  
**Trick**: Use DP to accumulate minimum costs.  
**Methodology**:
- Use `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

---

### **28. Word Break**
**Prompt**: Check if a string can be segmented into dictionary words.  
**Trick**: Use DP to track valid prefixes.  
**Methodology**:
- Use a boolean array to track possible segmentations.

---

### **29. Implement LRU Cache**
**Prompt**: Implement an LRU cache with O(1) operations.  
**Trick**: Use a combination of a dictionary and a doubly-linked list.  
**Methodology**:
- Evict least recently used keys using the list.

---

### **30. Kth Largest Element**
**Prompt**: Find the k-th largest element in an array.  
**Trick**: Use a min-heap of size `k`.  
**Methodology**:
- Push elements into the heap and pop the smallest when size exceeds `k`.

---

### Methodology
1. **Understand Patterns**: Most problems fall into repeatable categories (e.g., DP, two-pointer, sliding window).
2. **Analyze Complexity**: Evaluate time and space requirements for your solution.
3. **Optimize Iteratively**: Start with a brute-force solution and refine it.
4. **Learn Debugging**: Use print statements or tools to debug recursive and iterative solutions.

Let’s start solving one prompt at a time