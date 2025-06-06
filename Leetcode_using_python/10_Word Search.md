### **10. Word Search**  

#### **Prompt**  
**Goal**: Check if a given word exists in a 2D grid (board) by following adjacent letters. Letters must be connected horizontally or vertically, and the same cell cannot be reused.  

---

### **Trick**  
- Use **Depth-First Search (DFS)** with **backtracking** to explore all possible paths for forming the word.  

---

### **Methodology**  

1. **Define a Helper Function**:  
   - Use a recursive function to check if the current cell and its neighbors can form the word.  

2. **Base Cases**:  
   - If the word is empty (`len(word) == 0`), return `True` (success).  
   - If the current cell is out of bounds or already visited, or its letter does not match the first character of the word, return `False`.  

3. **Mark the Cell as Visited**:  
   - Temporarily mark the current cell to avoid revisiting during the same path.  

4. **Recursive Exploration**:  
   - Explore all four possible directions (up, down, left, right).  
   - If any direction completes the word, return `True`.  

5. **Backtracking**:  
   - Restore the cell to its original state after exploring all directions to allow other paths to use it.  

6. **Iterate Over the Grid**:  
   - Start the DFS from every cell to handle cases where the word can start from multiple locations.  

---

### **Python Implementation**  
```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index):
        # Base case: All characters in the word are found
        if index == len(word):
            return True
        
        # Out of bounds or mismatched character
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[index]:
            return False
        
        # Mark the current cell as visited
        temp = board[r][c]
        board[r][c] = "#"
        
        # Explore all directions (up, down, left, right)
        found = (dfs(r + 1, c, index + 1) or
                 dfs(r - 1, c, index + 1) or
                 dfs(r, c + 1, index + 1) or
                 dfs(r, c - 1, index + 1))
        
        # Backtrack: restore the cell's value
        board[r][c] = temp
        
        return found
    
    # Start DFS from each cell
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):  # Start searching for the word from cell (r, c)
                return True
    
    return False

# Example usage
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
word = "ABCCED"
print(word_search(board, word))  # Output: True
```

---

### **Key Points**  
1. **Time Complexity**:  
   - In the worst case, the algorithm explores all paths in the grid.  
   - Complexity: \( O(M \times N \times 4^L) \), where:  
     - \( M \) and \( N \) are the grid's dimensions.  
     - \( L \) is the length of the word.  
     - \( 4^L \): Four possible directions at each step for \( L \) steps.  

2. **Space Complexity**:  
   - \( O(L) \): Stack space for recursion, where \( L \) is the length of the word.  

3. **Edge Cases**:  
   - Empty grid or word: Return `False`.  
   - Word longer than total cells in the grid: Return `False`.  
   - Multiple starting points for the word: Check all cells.  

---

### **Example Walkthrough**  
**Input**:  
```plaintext
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
word = "SEE"
```

1. Start DFS from cell `(2, 2)`.  
2. Explore adjacent cells to match the next character (`E`).  
3. Successfully trace the path `(2, 2) → (2, 1) → (1, 1)`.  
4. Return `True`.