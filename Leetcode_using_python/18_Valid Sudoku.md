### **Valid Sudoku**

---

### **Prompt**  
**Goal**: Check if a given Sudoku board is valid. A valid Sudoku board follows these conditions:
1. Each row must contain the digits 1-9 without repetition.
2. Each column must contain the digits 1-9 without repetition.
3. Each 3x3 subgrid must contain the digits 1-9 without repetition.

---

### **Trick**  
- Use **hash sets** to keep track of seen numbers for rows, columns, and subgrids.  
- For each element on the board, check if it's already seen in its respective row, column, or subgrid.

---

### **Methodology**  

1. **Initialize Sets for Each Row, Column, and Subgrid**:  
   - Use three sets: one for rows, one for columns, and one for subgrids. Each set will store the numbers encountered so far for that row, column, or subgrid.

2. **Iterate Through the Board**:  
   - For each cell `(i, j)` in the board:
     - If the cell contains a number (between 1 and 9), check:
       - If the number already exists in the row's set, column's set, or subgrid's set.
       - If it does, the board is invalid.
       - Otherwise, add the number to the respective sets for that row, column, and subgrid.

3. **Subgrid Mapping**:  
   - To identify which 3x3 subgrid a cell belongs to, use the formula:  
     `subgrid_index = (i // 3) * 3 + (j // 3)`, which gives the index of the subgrid.

4. **Return True if No Violations**:  
   - If the entire board is traversed without finding any duplicates, return `True` (the board is valid).

---

### **Python Implementation**  

```python
def isValidSudoku(board):
    # Initialize sets for rows, columns, and subgrids
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    subgrids = [set() for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            
            if num == '.':
                continue  # Skip empty cells
            
            # Calculate the subgrid index
            subgrid_index = (i // 3) * 3 + (j // 3)
            
            # Check for duplicates in row, column, or subgrid
            if num in rows[i] or num in cols[j] or num in subgrids[subgrid_index]:
                return False
            
            # Add the number to the corresponding sets
            rows[i].add(num)
            cols[j].add(num)
            subgrids[subgrid_index].add(num)
    
    return True

# Example usage
board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]
print(isValidSudoku(board))  # Output: True
```

---

### **Key Points**  

1. **Time Complexity**:  
   - \( O(n^2) \), where \( n = 9 \) (since the board is always 9x9). Thus, the time complexity is effectively constant, \( O(1) \), for a fixed-size Sudoku board.

2. **Space Complexity**:  
   - \( O(n^2) \), for the sets used to track numbers in rows, columns, and subgrids.  

3. **Edge Cases**:  
   - Board with empty cells ('.') should be skipped.
   - Board with repeated numbers in rows, columns, or subgrids should return `False`.  

---

### **Example Walkthrough**  

**Input**:  

```python
board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]
```

1. **Check for each number**:  
   - The number `5` at `(0,0)` is added to the row set, column set, and subgrid set.  
   - The number `3` at `(0,1)` is added similarly.  
   - The number `7` at `(0,4)` is added similarly, and so on.  

2. **Result**:  
   - No duplicates found during the traversal, and all rules are satisfied.  
   - **Output**: `True`.

This approach efficiently checks the validity of the Sudoku board by ensuring there are no duplicate numbers in any row, column, or subgrid.