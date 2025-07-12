
---

### **Difficulty Level: Beginner**  
**Objective:** Learn Solidity syntax, basic smart contract structure, and simple functionalities.  

#### **Assignment 1: Hello World Contract**  
- Write a contract that stores and retrieves a string message (`Hello, World!`).  
- Include:  
  - A state variable to store the message.  
  - A function to update the message.  
  - A function to read the message.  

#### **Assignment 2: Simple Token (ERC-20 Basics)**  
- Create a basic token with:  
  - A fixed supply (e.g., 1,000,000 tokens).  
  - A function to transfer tokens between two addresses.  
  - A mapping to track balances.  

#### **Assignment 3: Voting System**  
- Build a contract where:  
  - The owner can add candidates.  
  - Users can vote for a candidate (one vote per address).  
  - A function displays the vote count for each candidate.  

---

### **Difficulty Level: Intermediate**  
**Objective:** Work with security best practices, inheritance, and more complex logic.  

#### **Assignment 4: Multi-Signature Wallet**  
- Create a wallet that requires **M-of-N** approvals for transactions.  
- Features:  
  - Add/remove owners.  
  - Submit, approve, and execute transactions.  
  - Prevent reentrancy attacks.  

#### **Assignment 5: Staking & Rewards**  
- Build a staking contract where:  
  - Users can deposit ETH/tokens.  
  - They earn rewards over time (e.g., 5% APR).  
  - Include withdrawal with penalties for early unstaking.  

#### **Assignment 6: Dutch Auction**  
- Implement a descending-price auction:  
  - The price starts high and decreases over time.  
  - Buyers can purchase at the current price.  
  - The auction ends when all items are sold or time expires.  

---

### **Difficulty Level: Advanced**  
**Objective:** Master gas optimization, complex DeFi logic, and security vulnerabilities.  

#### **Assignment 7: Decentralized Exchange (DEX)**  
- Build a simple AMM (Automated Market Maker) like Uniswap:  
  - Implement `addLiquidity`, `removeLiquidity`, and `swap` functions.  
  - Use the formula `x * y = k` for pricing.  
  - Handle LP (Liquidity Provider) tokens.  

#### **Assignment 8: Flash Loan Arbitrage**  
- Create a contract that:  
  - Takes a flash loan (e.g., from Aave or a mock pool).  
  - Executes an arbitrage between two DEXs.  
  - Repays the loan + fee in one transaction.  

#### **Assignment 9: Upgradeable Smart Contract**  
- Use **Proxy Patterns** (e.g., Transparent Proxy or UUPS):  
  - Deploy a logic contract and a proxy.  
  - Simulate upgrading the contract without losing state.  

---

### **Expert Level (Bonus Challenges)**  
- **Assignment 10:** Optimize a contract for **gas efficiency** (e.g., using assembly/Yul).  
- **Assignment 11:** Implement a **zk-SNARKs-based private transaction** system.  
- **Assignment 12:** Build a **DAO governance system** with voting and delegation.  

---
