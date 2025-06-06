### **House Robber Problem**

---

### **Prompt**  
**Goal**: Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police. The police will be alerted if two adjacent houses are robbed.

---

### **Trick**  
- **Dynamic Programming (DP)**:  
  - At each house, you have two choices:
    - **Skip** the current house and take the maximum amount robbed up to the previous house.
    - **Rob** the current house and add the amount robbed from the house two steps back (to avoid alerting the police).

---

### **Methodology**  
1. **State Representation**:  
   - Let `dp[i]` represent the maximum amount of money that can be robbed from the first `i` houses.
   
2. **Recurrence Relation**:  
   - The choice at each house `i` is either:
     - Skip house `i`, so the result is the same as `dp[i-1]`.
     - Rob house `i`, so the result is the amount robbed from house `i` (`nums[i]`) plus the amount robbed from house `i-2` (`dp[i-2]`).
   - The recurrence relation is:  
     \[
     dp[i] = \max(dp[i-1], dp[i-2] + nums[i])
     \]

3. **Base Case**:
   - `dp[0] = nums[0]` (if there’s only one house, rob it).
   - `dp[1] = max(nums[0], nums[1])` (for two houses, rob the one with the larger amount).

4. **Final Answer**:  
   - The final answer will be stored in `dp[n-1]`, where `n` is the number of houses.

---

### **Algorithm**:

1. Initialize a DP array `dp` where `dp[i]` stores the maximum amount of money robbed from the first `i` houses.
2. Set base cases for the first two houses.
3. Iterate through the list and apply the recurrence relation to fill up the DP table.
4. Return the value at `dp[n-1]` for the maximum amount that can be robbed.

---

### **Python Implementation**:

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        # Initialize DP array
        dp = [0] * len(nums)
        dp[0] = nums[0]  # Base case for the first house
        dp[1] = max(nums[0], nums[1])  # Base case for the second house
        
        # Fill DP array for remaining houses
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[-1]  # The answer is the maximum amount robbed from all houses

# Example usage
sol = Solution()
print(sol.rob([2, 3, 2]))  # Output: 3 (Rob house 1)
print(sol.rob([1, 2, 3, 1]))  # Output: 4 (Rob house 1 and house 3)
```

---

### **Explanation of the Code**:

1. **Edge Cases**:
   - If the input list is empty, return `0` because there are no houses to rob.
   - If there’s only one house, return the amount in that house.
   
2. **Base Cases**:
   - `dp[0] = nums[0]`: If there is only one house, rob it.
   - `dp[1] = max(nums[0], nums[1])`: For two houses, rob the one with the larger amount.

3. **Dynamic Programming**:
   - We iterate through each house starting from index `2` and compute `dp[i]` as the maximum of either skipping the current house (`dp[i-1]`) or robbing the current house (`dp[i-2] + nums[i]`).

4. **Final Result**:
   - After processing all the houses, `dp[len(nums) - 1]` will contain the maximum amount of money that can be robbed without alerting the police.

---

### **Time Complexity**:
- **O(n)**: We iterate through the list of houses once, and for each house, we perform constant-time operations.
  
### **Space Complexity**:
- **O(n)**: We use an array `dp` of size `n` to store the results for each house.

---

### **Space Optimization**:
If we want to reduce the space complexity, we can observe that we only need the last two values of `dp` at any point. Therefore, we can optimize the solution to use constant space:

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2, prev1 = nums[0], max(nums[0], nums[1])  # Base cases
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, current
        
        return prev1  # The maximum amount robbed from all houses

# Example usage
sol = Solution()
print(sol.rob([2, 3, 2]))  # Output: 3
print(sol.rob([1, 2, 3, 1]))  # Output: 4
```

In this optimized solution, we only use two variables (`prev1` and `prev2`) to track the last two results, reducing the space complexity to **O(1)**.

---

### **Example Walkthrough**:

**Example 1**:
- **Input**: `nums = [2, 3, 2]`
- **Steps**:
  - `dp[0] = 2`
  - `dp[1] = max(2, 3) = 3`
  - `dp[2] = max(3, 2 + 2) = 3`
- **Output**: `3`

**Example 2**:
- **Input**: `nums = [1, 2, 3, 1]`
- **Steps**:
  - `dp[0] = 1`
  - `dp[1] = max(1, 2) = 2`
  - `dp[2] = max(2, 1 + 3) = 4`
  - `dp[3] = max(4, 2 + 1) = 4`
- **Output**: `4`

---

### **Edge Cases**:
1. **Empty list**: If no houses are present, the maximum amount to rob is `0`.
2. **Single house**: If there is only one house, rob it.
3. **All houses with the same amount**: The DP approach will ensure that the maximum is calculated without alerting the police by skipping alternate houses.

---

This dynamic programming solution efficiently calculates the maximum money that can be robbed from a series of houses while avoiding adjacent house robberies, ensuring optimal performance even for large inputs.