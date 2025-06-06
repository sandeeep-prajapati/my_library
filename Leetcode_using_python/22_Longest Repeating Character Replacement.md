### **Longest Substring with Repeated Characters (Replace Characters)**

---

### **Prompt**  
**Goal**: Given a string, replace characters to get the longest substring that consists of only one repeated character.

---

### **Trick**  
- **Sliding Window with Frequency Map**:  
  - Use a sliding window to explore substrings.
  - Keep track of the frequency of characters in the window and adjust the window size based on the maximum frequency character.
  
---

### **Methodology**  
1. **Sliding Window**:
   - Use two pointers, `left` and `right`, to represent the sliding window.
   - Expand the window by moving the `right` pointer, and adjust the `left` pointer when the window becomes invalid (i.e., when the difference between the window size and the frequency of the most common character exceeds the allowed number of character replacements).

2. **Frequency Map**:
   - Maintain a frequency map (hashmap) to count occurrences of each character in the current window.
   - Track the character with the highest frequency in the current window.

3. **Adjust Window**:
   - If the number of characters that need to be replaced (i.e., the window size minus the frequency of the most frequent character) exceeds the allowed replacements, move the `left` pointer to reduce the window size.

4. **Calculate Maximum Length**:
   - Keep track of the maximum length of valid windows encountered.

---

### **Algorithm**:

1. Initialize a variable `max_count` to track the highest frequency of any character in the window.
2. Iterate over the string with the `right` pointer.
3. Update the frequency of the character at `right`.
4. If the window size (`right - left + 1`) minus `max_count` exceeds `k` (the number of allowed replacements), move the `left` pointer to shrink the window.
5. Update the result with the maximum window size found.

---

### **Python Implementation**:

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        left = 0
        max_length = 0
        frequency_map = {}
        max_count = 0
        
        for right in range(len(s)):
            # Update frequency map for the current character
            frequency_map[s[right]] = frequency_map.get(s[right], 0) + 1
            # Update the maximum frequency in the current window
            max_count = max(max_count, frequency_map[s[right]])
            
            # If the current window size minus the max frequency exceeds k, shrink the window
            if (right - left + 1) - max_count > k:
                frequency_map[s[left]] -= 1
                left += 1
                
            # Update the maximum length of valid window
            max_length = max(max_length, right - left + 1)
        
        return max_length

# Example usage
sol = Solution()
print(sol.characterReplacement("AABABBA", 1))  # Output: 4
```

---

### **Explanation of the Code**:

1. **Initialization**:
   - `left` is the left pointer of the sliding window.
   - `max_length` stores the maximum length of valid substrings found.
   - `frequency_map` tracks the frequency of characters in the window.
   - `max_count` tracks the frequency of the most common character in the window.

2. **Sliding Window**:
   - For each `right` pointer position, we expand the window to the right.
   - Update the frequency map and `max_count` accordingly.
   - If the difference between the window size and `max_count` exceeds `k`, we shrink the window from the left by moving the `left` pointer.

3. **Result**:
   - After processing all the characters in the string, the `max_length` will hold the length of the longest valid substring.

---

### **Time Complexity**:
- **O(n)**:  
  - We iterate through the string once with the `right` pointer, and the `left` pointer moves only when necessary, resulting in a linear time complexity.

### **Space Complexity**:
- **O(1)**:  
  - The frequency map has at most 26 entries (for each letter of the alphabet), so space complexity is constant.

---

### **Example Walkthrough**:

**Example 1**:
- **Input**:  
  `s = "AABABBA"`, `k = 1`
- **Steps**:
  - **Window 1**: `"AA"` (max frequency = 2), valid window length = 2
  - **Window 2**: `"AAB"` (max frequency = 2), valid window length = 3
  - **Window 3**: `"AABA"` (max frequency = 2), valid window length = 4
  - **Window 4**: `"AABAB"` (max frequency = 2), valid window length = 4
  - **Window 5**: `"AABABB"` (max frequency = 3), valid window length = 4
  - The longest valid window is `"AABAB"`, so the result is 4.

**Example 2**:
- **Input**:  
  `s = "ABAB", k = 2`
- **Steps**:
  - **Window 1**: `"AB"` (max frequency = 1), valid window length = 2
  - **Window 2**: `"ABA"` (max frequency = 2), valid window length = 3
  - **Window 3**: `"ABAB"` (max frequency = 2), valid window length = 4
  - The longest valid window is `"ABAB"`, so the result is 4.

---

This approach efficiently calculates the longest substring by leveraging a sliding window with a frequency map and adjusting the window size based on the character frequency, ensuring a linear-time solution.