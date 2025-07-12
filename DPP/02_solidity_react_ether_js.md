
---

## **Difficulty Level: Beginner**  
**Objective:** Learn basic smart contract interaction with a frontend.  

### **1. Simple Wallet Connection**  
**Task:**  
- Build a React app that connects to MetaMask using Ethers.js.  
- Display the connected wallet address and ETH balance.  

**Skills Learned:**  
- Ethers.js `provider` and `signer`.  
- React state management for wallet connection.  

---

### **2. Token Balance Checker**  
**Task:**  
- Deploy a simple ERC-20 token contract (or use an existing one like DAI).  
- Create a React app where users can:  
  - Enter a token contract address.  
  - Enter their wallet address.  
  - Fetch and display their token balance.  

**Skills Learned:**  
- Interacting with ERC-20 contracts (`balanceOf`).  
- Basic Ethers.js contract calls.  

---

### **3. Basic Transaction Sender**  
**Task:**  
- Build a React app where users can:  
  - Enter an ETH amount and recipient address.  
  - Send ETH transactions via MetaMask.  
  - Display transaction status (pending, success, failure).  

**Skills Learned:**  
- Sending transactions with Ethers.js (`sendTransaction`).  
- Handling transaction receipts.  

---

## **Difficulty Level: Intermediate**  
**Objective:** Work with more complex smart contracts and frontend state management.  

### **4. Todo List dApp (Stateful Smart Contract)**  
**Task:**  
- Create a Solidity contract that stores tasks (`addTask`, `toggleComplete`, `deleteTask`).  
- Build a React frontend to:  
  - Fetch and display tasks for the connected wallet.  
  - Allow adding, toggling, and deleting tasks.  

**Skills Learned:**  
- Reading and writing to a smart contract.  
- React state updates based on blockchain data.  

---

### **5. Staking dApp (with Rewards)**  
**Task:**  
- Write a Solidity staking contract where users can:  
  - Deposit ERC-20 tokens.  
  - Earn rewards over time.  
  - Withdraw staked tokens + rewards.  
- Build a React frontend to interact with it.  

**Skills Learned:**  
- Time-based rewards logic in Solidity.  
- Tracking staking positions in the frontend.  

---

### **6. Multi-Signature Wallet (Advanced Transactions)**  
**Task:**  
- Create a multi-sig wallet contract (requires `N` out of `M` approvals).  
- Build a React dashboard where owners can:  
  - Propose transactions.  
  - Approve/reject transactions.  
  - Execute approved transactions.  

**Skills Learned:**  
- Complex smart contract logic (approvals, execution).  
- Frontend for multi-step transactions.  

---

## **Difficulty Level: Advanced**  
**Objective:** Build full-fledged DeFi or DAO applications.  

### **7. Decentralized Exchange (DEX) like Uniswap**  
**Task:**  
- Implement a Solidity AMM (Automated Market Maker) with:  
  - `addLiquidity`, `removeLiquidity`, `swap` functions.  
  - LP token minting/burning.  
- Build a React UI for swapping tokens and adding liquidity.  

**Skills Learned:**  
- AMM math (`x * y = k`).  
- Advanced Ethers.js interactions.  

---

### **8. DAO Governance dApp (Voting & Proposals)**  
**Task:**  
- Create a Solidity DAO contract with:  
  - Proposal creation.  
  - Voting with governance tokens.  
  - Execution of passed proposals.  
- Build a React frontend to submit/vote on proposals.  

**Skills Learned:**  
- Token-weighted voting.  
- Handling complex governance logic.  

---

### **9. NFT Marketplace (Minting, Buying, Selling)**  
**Task:**  
- Deploy an ERC-721 NFT contract with a marketplace.  
- Features:  
  - Mint NFTs.  
  - List NFTs for sale.  
  - Buy NFTs with ETH.  
- React frontend to browse and trade NFTs.  

**Skills Learned:**  
- NFT standards (ERC-721).  
- Marketplace escrow logic.  

---

## **Bonus: Expert Challenges**  
- **10. Flash Loan Arbitrage Bot** (React dashboard + Solidity executor).  
- **11. Layer-2 dApp** (Using Optimism/Arbitrum + Ethers.js).  
- **12. Gasless Transactions** (Meta-transactions with EIP-2771).  

---

### **Tools to Use:**  
- **Smart Contracts:** Solidity + Hardhat/Foundry.  
- **Frontend:** React + Ethers.js + Vite.  
- **Styling:** TailwindCSS or Chakra UI.  
- **Testing:** Hardhat tests + React testing library.  
