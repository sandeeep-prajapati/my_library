In **ethers.js**, you can fetch **block details**, **transaction data**, and **historical balances** using a **provider** (like `JsonRpcProvider`, `AlchemyProvider`, or `InfuraProvider`). Below are examples for each task:

---

## **1. Fetching Block Details**
To get block information (number, hash, timestamp, transactions, etc.):

### **Example: Get Latest Block**
```javascript
const { ethers } = require("ethers");

const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL"); // e.g., Infura, Alchemy

async function getLatestBlock() {
  const block = await provider.getBlock("latest"); // "latest" or block number
  console.log(block);
}

getLatestBlock();
```

### **Output:**
```json
{
  "hash": "0x...",
  "number": 19237842,
  "timestamp": 1656789000,
  "transactions": ["0x...", "0x..."],
  "parentHash": "0x...",
  "miner": "0x...",
  "gasLimit": 30000000,
  "gasUsed": 15000000
}
```

---

## **2. Fetching Transaction Data**
To fetch details of a specific transaction (sender, receiver, value, gas, etc.):

### **Example: Get Transaction Details**
```javascript
async function getTransaction(txHash) {
  const tx = await provider.getTransaction(txHash);
  console.log(tx);
}

getTransaction("0x..."); // Replace with a real TX hash
```

### **Output:**
```json
{
  "hash": "0x...",
  "blockNumber": 19237842,
  "from": "0x...",
  "to": "0x...",
  "value": ethers.utils.parseEther("1.0"),
  "gasPrice": ethers.BigNumber.from("20000000000"),
  "gasLimit": 21000,
  "data": "0x..."
}
```

---

## **3. Fetching Historical Balances**
To check an address's balance at a specific block (historical balance):

### **Example: Get Balance at a Past Block**
```javascript
async function getHistoricalBalance(address, blockNumber) {
  const balance = await provider.getBalance(address, blockNumber);
  console.log(`Balance at block ${blockNumber}: ${ethers.formatEther(balance)} ETH`);
}

getHistoricalBalance("0x...", 15000000); // Address + block number
```

### **Output:**
```
Balance at block 15000000: 2.5 ETH
```

---

## **4. Fetching All Transactions for an Address**
If you need **all transactions** sent/received by an address, you typically need an **Etherscan-like API** (ethers.js alone doesn't support this directly).

### **Example: Using Etherscan API with ethers.js**
```javascript
const { ethers } = require("ethers");
const axios = require("axios");

const ETHERSCAN_API_KEY = "YOUR_API_KEY";
const address = "0x...";

async function getAddressTransactions() {
  const url = `https://api.etherscan.io/api?module=account&action=txlist&address=${address}&startblock=0&endblock=99999999&sort=asc&apikey=${ETHERSCAN_API_KEY}`;
  const response = await axios.get(url);
  console.log(response.data.result);
}

getAddressTransactions();
```

### **Output:**
```json
[
  {
    "blockNumber": "123456",
    "timeStamp": "1625097600",
    "hash": "0x...",
    "from": "0x...",
    "to": "0x...",
    "value": "1000000000000000000",
    "gas": "21000",
    "gasPrice": "20000000000"
  },
  ...
]
```

---

## **Summary**
| Task | Method |
|------|--------|
| **Get Block Details** | `provider.getBlock(blockNumber)` |
| **Get Transaction** | `provider.getTransaction(txHash)` |
| **Get Historical Balance** | `provider.getBalance(address, blockNumber)` |
| **Get All Transactions** | Requires **Etherscan API** (ethers.js alone doesn't support it) |

### **Recommended Providers**
- **Free RPCs**: [Alchemy](https://www.alchemy.com/), [Infura](https://infura.io/), [QuickNode](https://www.quicknode.com/)
- **Etherscan API**: For transaction history.

Would you like help setting up a specific provider (like Alchemy or Infura)? 🚀