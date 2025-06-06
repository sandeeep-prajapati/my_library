### **Decode Ways**

---

### **Prompt**  
**Goal**: Given a string of digits, decode it into possible letter combinations. Each digit (or pair of digits) can correspond to a letter from 'A' to 'Z' (1 = 'A', 2 = 'B', ..., 26 = 'Z'). Determine how many possible decodings there are for the given string.

---

### **Trick**  
- **Dynamic Programming (DP)**:  
  - Use DP to count the number of ways to decode substrings progressively.
  - Use the constraints (1–26) to decide valid character mappings.

---

### **Methodology**  
1. **Dynamic Programming Setup**:
   - Let `dp[i]` represent the number of ways to decode the substring from the start up to the `i`-th character.
   - Initialize `dp[0] = 1` because there’s one way to decode an empty string.
   
2. **Iterate Through the String**:
   - For each position `i` in the string, consider two cases:
     - **Single digit decoding**: If the current digit forms a valid letter (i.e., it's between '1' and '9'), add `dp[i-1]` to `dp[i]` (since it's the number of ways to decode the previous part).
     - **Two digit decoding**: If the last two digits form a valid letter (i.e., between '10' and '26'), add `dp[i-2]` to `dp[i]`.
   
3. **Constraints**:
   - Skip cases where the string has invalid digits (e.g., '0' at the beginning of a pair, or digits larger than '26').
   - Ensure proper handling of edge cases such as strings starting with '0' or containing '0' at invalid places.

4. **Final Result**:
   - The final result will be stored in `dp[n]`, where `n` is the length of the string.

---

### **Algorithm**:

1. Initialize a DP array `dp` of size `n + 1` where `n` is the length of the input string.
2. Set `dp[0] = 1` because there is one way to decode an empty string.
3. Iterate through the string and update `dp[i]` based on valid decodings of single and two-digit numbers.
4. Return `dp[n]` which gives the total number of decodings for the string.

---

### **Python Implementation**:

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':  # Early exit for empty string or strings starting with '0'
            return 0
        
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1  # There is one way to decode an empty string
        
        for i in range(1, n + 1):
            # Single digit decoding (1 to 9)
            if s[i-1] != '0':  # '0' cannot stand alone as a valid digit
                dp[i] += dp[i-1]
            
            # Two digit decoding (10 to 26)
            if i > 1 and s[i-2] == '1' or (s[i-2] == '2' and s[i-1] <= '6'):
                dp[i] += dp[i-2]
        
        return dp[n]

# Example usage
sol = Solution()
print(sol.numDecodings("12"))  # Output: 2 (can be "AB" or "L")
print(sol.numDecodings("226"))  # Output: 3 ("BBF", "BZ", "VF")
print(sol.numDecodings("0"))  # Output: 0 (invalid input)
```

---

### **Explanation of the Code**:

1. **Base Case**:
   - If the string is empty or starts with a '0', there are no valid decodings, so we return 0.

2. **DP Array Initialization**:
   - `dp[i]` represents the number of ways to decode the string `s[:i]`.
   - We initialize `dp[0] = 1` because there is one way to decode an empty string.

3. **Iterating Through the String**:
   - For each index `i` in the string, check if the current character forms a valid single digit (1–9).
   - If `i > 1`, also check if the last two characters form a valid two-digit number (10–26).
   
4. **Updating DP**:
   - For each valid decoding, add the corresponding values from previous positions in the `dp` array.
   
5. **Final Answer**:
   - After processing the entire string, `dp[n]` will contain the total number of ways to decode the string.

---

### **Time Complexity**:
- **O(n)**:  
  - We iterate over the string once, and for each character, we perform constant-time operations (checking the current and previous characters).

### **Space Complexity**:
- **O(n)**:  
  - We use a DP array of size `n + 1` to store the number of ways to decode each substring.

---

### **Example Walkthrough**:

**Example 1**:
- **Input**: `s = "12"`
- **Steps**:
  - `dp[0] = 1` (one way to decode an empty string)
  - `dp[1] = 1` (only "A" for "1")
  - `dp[2] = 2` (either "AB" or "L" for "12")
- **Output**: `2`

**Example 2**:
- **Input**: `s = "226"`
- **Steps**:
  - `dp[0] = 1`
  - `dp[1] = 1` ("B")
  - `dp[2] = 2` ("BB", "Z")
  - `dp[3] = 3` ("BBF", "BZ", "VF")
- **Output**: `3`

**Example 3**:
- **Input**: `s = "0"`
- **Steps**: Since "0" can't decode to any letter, the output is `0`.

---

### **Edge Cases**:
1. **Leading '0'**: Any string starting with '0' should return `0` because '0' cannot represent a letter on its own.
2. **Empty String**: An empty string should return `1` as there is one way to decode nothing.
3. **String with Invalid '0' Pairs**: A string like "30" should return `0` because '30' is not a valid mapping.

---

This dynamic programming approach efficiently computes the number of valid decodings by progressively building solutions for substrings and using previously computed values to optimize the solution.