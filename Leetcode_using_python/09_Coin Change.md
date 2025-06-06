### **9. Coin Change**  

#### **Prompt**  
**Goal**: Given an amount and a list of coins, find the minimum number of coins required to make the amount.  

---

### **Trick**  
- Use **Dynamic Programming (DP)** with a **bottom-up** approach to solve subproblems iteratively.  

---

### **Methodology**  

1. **Define a DP Table**:  
   - `dp[i]` represents the minimum number of coins required to make the amount `i`.  

2. **Initialize the DP Table**:  
   - `dp[0] = 0` because no coins are needed to make the amount `0`.  
   - For all other amounts, initialize `dp[i] = inf` (infinity) to signify that the amount is initially unreachable.  

3. **Update the DP Table**:  
   - For each coin in the list of coins, update the table for amounts that can be formed using that coin.  
   - Formula:  
     \[
     dp[i] = \min(dp[i], 1 + dp[i - \text{coin}])
     \]  
     where `1 + dp[i - coin]` accounts for adding the current coin to the solution for the remaining amount (`i - coin`).  

4. **Final Result**:  
   - If `dp[amount]` is still `inf`, return `-1` (not possible to make the amount).  
   - Otherwise, return `dp[amount]`.  

---

### **Python Implementation**  
```python
def coin_change(coins, amount):
    # Initialize DP table
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins to make amount 0
    
    # Fill the DP table
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], 1 + dp[i - coin])
    
    # Return result
    return dp[amount] if dp[amount] != float('inf') else -1

# Example usage
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))  # Output: 3 (5 + 5 + 1)
```

---

### **Key Points**  
1. **Time Complexity**:  
   - \( O(n \times \text{amount}) \), where \( n \) is the number of coins.  
   - For each coin, iterate through the DP table up to `amount`.  

2. **Space Complexity**:  
   - \( O(\text{amount}) \), as only a single DP table of size `amount + 1` is used.  

3. **Edge Cases**:  
   - Amount is `0`: Always return `0` as no coins are needed.  
   - Coins list is empty: Return `-1` unless the amount is `0`.  
   - Amount cannot be formed with the given coins: Return `-1`.  

---

### **Example Walkthrough**  
**Input**: `coins = [1, 2, 5]`, `amount = 11`  

1. **Initialization**:  
   \[
   dp = [0, \infty, \infty, \infty, \infty, \infty, \infty, \infty, \infty, \infty, \infty, \infty]
   \]

2. **Process Coin = 1**:  
   Update all amounts from `1` to `11`:  
   \[
   dp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
   \]

3. **Process Coin = 2**:  
   Update all amounts from `2` to `11`:  
   \[
   dp = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
   \]

4. **Process Coin = 5**:  
   Update all amounts from `5` to `11`:  
   \[
   dp = [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3]
   \]

5. **Result**: `dp[11] = 3` (minimum coins: \( 5 + 5 + 1 \)).