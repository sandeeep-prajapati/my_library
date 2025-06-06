### **Word Break Problem**

---

### **Prompt**  
**Goal**: Given a string `s` and a dictionary of words `wordDict`, check if `s` can be segmented into a space-separated sequence of dictionary words. 

---

### **Trick**  
- **Dynamic Programming (DP)**: Use a DP array to track which substrings of the string can be segmented into words from the dictionary.

---

### **Methodology**  
1. **Initialization**:
   - Use a boolean DP array `dp` of length `len(s) + 1` where `dp[i]` represents whether the substring `s[0:i]` can be segmented into words from the dictionary.
   - Initially, `dp[0]` is `True` because an empty string can trivially be segmented.

2. **Recurrence Relation**:
   - For each index `i` from 1 to `len(s)`, check if there exists a `j` such that:
     - `dp[j]` is `True` (the substring `s[0:j]` can be segmented),
     - The substring `s[j:i]` is a word in the dictionary.
   - If such a `j` exists, then `dp[i]` should be `True`.

3. **Final Result**:
   - The final answer will be stored in `dp[len(s)]`. If it's `True`, then the string can be segmented, otherwise it cannot.

4. **Optimization**:
   - You can use a set for the dictionary to achieve O(1) lookups for each substring check.

---

### **Algorithm**:

1. **Step 1**: Initialize the DP array and set the base case.
2. **Step 2**: Iterate through each position in the string and check possible valid segmentations.
3. **Step 3**: Return the result stored in `dp[len(s)]`.

---

### **Python Implementation**:

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        # Step 1: Initialize the DP array
        dp = [False] * (len(s) + 1)
        dp[0] = True  # Base case: empty string is always "segmented"
        
        # Step 2: Convert wordDict to a set for faster lookups
        wordSet = set(wordDict)
        
        # Step 3: Fill the DP array
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        
        # Step 4: Return the result for the entire string
        return dp[len(s)]

# Example usage
sol = Solution()
print(sol.wordBreak("leetcode", ["leet", "code"]))  # Output: True
```

---

### **Explanation of the Code**:

1. **Initialization**:
   - We initialize the `dp` array with `False` values, except for `dp[0]` which is `True` because the empty string can always be segmented.
   
2. **Dictionary Conversion**:
   - We convert `wordDict` into a set (`wordSet`) for faster lookup when checking whether a substring exists in the dictionary.

3. **DP Array Calculation**:
   - For each index `i` in the string, we iterate through possible indices `j` from `0` to `i`. For each pair of indices `j` and `i`, we check if:
     - `dp[j]` is `True` (the substring `s[0:j]` can be segmented),
     - The substring `s[j:i]` is in the dictionary (`wordSet`).
   - If both conditions are true, it means the substring `s[0:i]` can be segmented, so we set `dp[i] = True`.

4. **Final Result**:
   - The value `dp[len(s)]` holds the final answer, indicating whether the entire string can be segmented into dictionary words.

---

### **Time Complexity**:
- **O(n^2)**: We have a double loop:
  - Outer loop runs for each character in the string (`n` iterations).
  - Inner loop checks every substring ending at index `i` (up to `i` iterations).
  
- **Space Complexity**:
  - **O(n)**: The DP array has a length of `n + 1`.

---

### **Example Walkthrough**:

**Example 1**:
- **Input**: `s = "leetcode", wordDict = ["leet", "code"]`
  - `dp[0] = True` (base case: empty string)
  - For `i = 4`, we find that the substring `s[0:4] = "leet"` is in the dictionary, so `dp[4] = True`.
  - For `i = 8`, we find that the substring `s[4:8] = "code"` is in the dictionary, so `dp[8] = True`.
  - Final output: `dp[len(s)] = dp[8] = True`.

- **Output**: `True`

**Example 2**:
- **Input**: `s = "applepenapple", wordDict = ["apple", "pen"]`
  - `dp[0] = True` (base case: empty string)
  - For `i = 5`, we find that `s[0:5] = "apple"` is in the dictionary, so `dp[5] = True`.
  - For `i = 10`, we find that `s[5:10] = "pen"` is in the dictionary, so `dp[10] = True`.
  - For `i = 15`, we find that `s[10:15] = "apple"` is in the dictionary, so `dp[15] = True`.
  - Final output: `dp[len(s)] = dp[15] = True`.

- **Output**: `True`

---

### **Edge Cases**:

1. **Empty String**:
   - If `s` is an empty string, the result is always `True` because an empty string can trivially be segmented.

2. **No Matching Segments**:
   - If no valid segmentation exists, the result will be `False`.

3. **Dictionary Contains Entire String**:
   - If the entire string itself is in the dictionary, the result will be `True`.

4. **Large Strings**:
   - The solution is efficient enough to handle moderately large strings due to the quadratic time complexity. For extremely large strings, further optimizations or techniques like Trie-based dynamic programming might be considered.