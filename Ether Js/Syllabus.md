#### **01_Introduction_to_ethers.js.md**
   - **Topic:** What is `ethers.js`, and why is it a crucial library for Ethereum development?

#### **02_Setting_Up_ethers.js.md**
   - **Topic:** How to install and set up `ethers.js` in a Node.js or browser environment.

#### **03_Connecting_to_Ethereum_Networks.md**
   - **Topic:** How to connect to Ethereum networks (mainnet, testnets) using `ethers.js` providers.

#### **04_Wallet_Management_with_ethers.js.md**
   - **Topic:** Creating, importing, and managing Ethereum wallets using `ethers.Wallet`.

#### **05_Sending_ETH_Transactions.md**
   - **Topic:** How to send ETH transactions between wallets using `ethers.js`.

#### **06_Interacting_with_Smart_Contracts.md**
   - **Topic:** Connecting to and interacting with smart contracts using `ethers.js` and ABI.

#### **07_Reading_Contract_State.md**
   - **Topic:** Querying and reading data from smart contracts (e.g., token balances, storage values).

#### **08_Writing_to_Smart_Contracts.md**
   - **Topic:** Sending transactions to update the state of a smart contract (e.g., calling `set` functions).

#### **09_Listening_to_Contract_Events.md**
   - **Topic:** How to listen for and handle events emitted by smart contracts.

#### **10_Deploying_Smart_Contracts.md**
   - **Topic:** Deploying a smart contract to the Ethereum network using `ethers.js`.

#### **11_Gas_Management.md**
   - **Topic:** Estimating gas limits, setting gas prices, and optimizing transaction costs.

#### **12_Signing_and_Verifying_Messages.md**
   - **Topic:** Signing messages with a wallet and verifying signatures programmatically.

#### **13_Handling_Transaction_Receipts.md**
   - **Topic:** Fetching and interpreting transaction receipts to confirm successful transactions.

#### **14_Nonce_Management.md**
   - **Topic:** Managing transaction nonces to handle multiple transactions sequentially.

#### **15_Working_with_ENS.md**
   - **Topic:** Resolving Ethereum Name Service (ENS) domains and reverse lookups using `ethers.js`.

#### **16_Querying_Blockchain_Data.md**
   - **Topic:** Fetching block details, transaction data, and historical balances.

#### **17_Switching_Networks.md**
   - **Topic:** Switching between Ethereum networks (mainnet, testnets) dynamically.

#### **18_Error_Handling_in_ethers.js.md**
   - **Topic:** Implementing robust error handling for common `ethers.js` scenarios.

#### **19_Advanced_Features.md**
   - **Topic:** Exploring advanced features like multicall, batch transactions, and custom providers.

#### **20_Testing_and_Debugging.md**
   - **Topic:** Writing unit tests and debugging `ethers.js` scripts effectively.

---
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
---
# **Switching Between Ethereum Networks Dynamically in ethers.js**

To dynamically switch between **Ethereum Mainnet** and **testnets** (like **Goerli, Sepolia, Arbitrum, Optimism, etc.**) in **ethers.js**, you can use:

1. **Predefined `ethers` Providers** (for well-known networks).
2. **Custom RPC URLs** (for any EVM-compatible chain).
3. **Dynamic Provider Switching** (at runtime).

---

## **1. Using Predefined `ethers` Networks**
`ethers.js` has built-in support for common networks:

```javascript
const { ethers } = require("ethers");

// Built-in networks
const networks = {
  mainnet: ethers.getDefaultProvider("mainnet"),
  goerli: ethers.getDefaultProvider("goerli"),
  sepolia: ethers.getDefaultProvider("sepolia"),
  arbitrum: ethers.getDefaultProvider("arbitrum"),
  optimism: ethers.getDefaultProvider("optimism"),
};

// Example: Switch to Goerli
const provider = networks.goerli;
```

### **Supported Default Networks**
| Network | Chain ID | Provider Alias |
|---------|---------|----------------|
| Ethereum Mainnet | 1 | `"mainnet"` |
| Goerli (deprecated) | 5 | `"goerli"` |
| Sepolia | 11155111 | `"sepolia"` |
| Arbitrum One | 42161 | `"arbitrum"` |
| Optimism | 10 | `"optimism"` |

⚠️ **Note:** `ethers.getDefaultProvider()` uses free public endpoints (limited rate). For production, use **Alchemy/Infura/QuickNode**.

---

## **2. Using Custom RPC URLs (Recommended)**
For full control, use a **custom `JsonRpcProvider`**:

```javascript
const { ethers } = require("ethers");

const RPC_URLS = {
  mainnet: "https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
  goerli: "https://eth-goerli.g.alchemy.com/v2/YOUR_API_KEY",
  sepolia: "https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY",
  arbitrum: "https://arb-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
  optimism: "https://opt-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
};

function getProvider(network) {
  return new ethers.JsonRpcProvider(RPC_URLS[network]);
}

// Example: Switch to Sepolia
const provider = getProvider("sepolia");
```

---

## **3. Dynamic Network Switching (Full Example)**
Here’s a complete example with **dynamic network switching**:

```javascript
const { ethers } = require("ethers");

// Define RPC endpoints
const NETWORKS = {
  mainnet: {
    name: "Ethereum Mainnet",
    rpc: "https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
    chainId: 1,
  },
  sepolia: {
    name: "Sepolia Testnet",
    rpc: "https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY",
    chainId: 11155111,
  },
  arbitrum: {
    name: "Arbitrum One",
    rpc: "https://arb-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
    chainId: 42161,
  },
};

// Get provider for a network
function getProvider(networkKey) {
  if (!NETWORKS[networkKey]) {
    throw new Error(`Unknown network: ${networkKey}`);
  }
  return new ethers.JsonRpcProvider(NETWORKS[networkKey].rpc);
}

// Example: Fetch balance on different networks
async function fetchBalance(address, networkKey) {
  const provider = getProvider(networkKey);
  const balance = await provider.getBalance(address);
  console.log(
    `Balance on ${NETWORKS[networkKey].name}: ${ethers.formatEther(balance)} ETH`
  );
}

fetchBalance("0x...", "mainnet"); // Mainnet
fetchBalance("0x...", "sepolia"); // Sepolia
```

---

## **4. Using `ethers.Network` for Chain Metadata**
You can also use `ethers.Network` to handle chain metadata:

```javascript
const { ethers } = require("ethers");

const customNetwork = {
  name: "My Custom Network",
  chainId: 1234,
};

const provider = new ethers.JsonRpcProvider("https://my.rpc.url", customNetwork);
```

---

## **5. Switching Networks in a Frontend (MetaMask)**
If you're working in a **browser environment** (like with MetaMask), you can prompt the user to switch networks:

```javascript
async function switchNetwork(chainId) {
  if (!window.ethereum) throw new Error("MetaMask not installed!");

  await window.ethereum.request({
    method: "wallet_switchEthereumChain",
    params: [{ chainId: `0x${chainId.toString(16)}` }],
  });
}

// Example: Switch to Sepolia (Chain ID: 11155111)
switchNetwork(11155111);
```

### **Common Chain IDs**
| Network | Chain ID (Decimal) | Chain ID (Hex) |
|---------|-------------------|----------------|
| Ethereum Mainnet | 1 | `0x1` |
| Goerli | 5 | `0x5` |
| Sepolia | 11155111 | `0xAA36A7` |
| Arbitrum | 42161 | `0xA4B1` |
| Optimism | 10 | `0xA` |

---

## **Summary**
| Method | Use Case |
|--------|----------|
| **`ethers.getDefaultProvider()`** | Quick testing (public RPCs) |
| **Custom `JsonRpcProvider`** | Best for production (Alchemy/Infura) |
| **`ethers.Network`** | Custom EVM chains |
| **MetaMask `wallet_switchEthereumChain`** | Frontend network switching |
---
# **Robust Error Handling in `ethers.js` for Common Scenarios**

When working with `ethers.js`, network requests, transactions, and contract interactions can fail for various reasons (e.g., RPC errors, gas issues, invalid inputs). Below are **best practices for error handling** in different scenarios.

---

## **1. General Error Handling Structure**
Wrap `ethers.js` operations in `try-catch` blocks to gracefully handle failures.

```javascript
const { ethers } = require("ethers");

async function safeEthersCall() {
  try {
    const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
    const balance = await provider.getBalance("0x...");
    console.log(`Balance: ${ethers.formatEther(balance)} ETH`);
  } catch (error) {
    console.error("Error in ethers operation:", error.message);
    // Handle retries, fallback RPC, or user feedback
  }
}

safeEthersCall();
```

---

## **2. Common Error Scenarios & Fixes**

### **🔹 1. Provider Connection Failures**
**Causes:**  
- Invalid RPC URL  
- Network downtime  
- Rate limiting  

**Handling:**  
```javascript
async function getProvider() {
  const RPC_URLS = [
    "https://mainnet.infura.io/v3/YOUR_KEY", // Primary
    "https://eth.llamarpc.com", // Fallback
  ];

  for (const url of RPC_URLS) {
    try {
      const provider = new ethers.JsonRpcProvider(url);
      await provider.getBlockNumber(); // Test connection
      return provider;
    } catch (error) {
      console.warn(`Failed RPC (${url}):`, error.message);
    }
  }
  throw new Error("All RPCs failed");
}
```

---

### **🔹 2. Transaction Reverts (e.g., `insufficient funds`, `reverted`)**
**Causes:**  
- Insufficient gas  
- Smart contract revert  
- Incorrect parameters  

**Handling:**  
```javascript
async function sendTransaction() {
  const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
  const signer = new ethers.Wallet("PRIVATE_KEY", provider);

  try {
    const tx = await signer.sendTransaction({
      to: "0x...",
      value: ethers.parseEther("1.0"),
    });
    await tx.wait(); // Wait for confirmation
    console.log("Tx successful:", tx.hash);
  } catch (error) {
    if (error.code === "INSUFFICIENT_FUNDS") {
      console.error("Insufficient balance for gas + value");
    } else if (error.reason === "transaction failed") {
      console.error("Tx reverted:", error.receipt?.transactionHash);
    } else {
      console.error("Tx error:", error.message);
    }
  }
}
```

---

### **🔹 3. Contract Call Failures**
**Causes:**  
- Incorrect ABI  
- Reverted calls  
- Invalid function args  

**Handling:**  
```javascript
const contractABI = [...];
const contractAddress = "0x...";

async function callContract() {
  const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
  const contract = new ethers.Contract(contractAddress, contractABI, provider);

  try {
    const data = await contract.someFunction();
    console.log("Result:", data);
  } catch (error) {
    if (error.code === "CALL_EXCEPTION") {
      console.error("Contract reverted:", error.reason);
    } else if (error.code === "INVALID_ARGUMENT") {
      console.error("Invalid function args");
    } else {
      console.error("Contract error:", error.message);
    }
  }
}
```

---

### **🔹 4. Gas Estimation Failures**
**Causes:**  
- Gas too low  
- Complex reverts  

**Handling:**  
```javascript
async function estimateGasSafe() {
  const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
  const signer = new ethers.Wallet("PRIVATE_KEY", provider);

  try {
    const gasEstimate = await provider.estimateGas({
      from: signer.address,
      to: "0x...",
      value: ethers.parseEther("0.1"),
    });
    console.log("Gas estimate:", gasEstimate.toString());
  } catch (error) {
    console.error("Gas estimation failed:", error.reason || error.message);
  }
}
```

---

### **🔹 5. Wallet/Private Key Errors**
**Causes:**  
- Invalid private key  
- Incorrect mnemonic  

**Handling:**  
```javascript
function loadWalletSafely(privateKey) {
  try {
    return new ethers.Wallet(privateKey);
  } catch (error) {
    if (error.code === "INVALID_ARGUMENT") {
      throw new Error("Invalid private key/mnemonic");
    }
    throw error;
  }
}
```

---

## **3. Advanced: Error Classification**
Use `error.code` or `error.reason` to categorize failures:

| **Error Code**          | **Meaning**                          | **Solution**                      |
|-------------------------|--------------------------------------|-----------------------------------|
| `NETWORK_ERROR`         | RPC connection failed                | Retry or fallback RPC             |
| `CALL_EXCEPTION`        | Contract reverted                    | Check contract logic              |
| `INSUFFICIENT_FUNDS`    | Not enough ETH for gas + value       | Top up wallet                     |
| `UNPREDICTABLE_GAS_LIMIT` | Gas estimation failed               | Manually set `gasLimit`           |
| `INVALID_ARGUMENT`      | Wrong input format                   | Validate inputs before sending    |

---

## **4. Best Practices**
1. **Retry Logic** (for transient errors like RPC timeouts):
   ```javascript
   async function withRetry(fn, maxRetries = 3) {
     for (let i = 0; i < maxRetries; i++) {
       try {
         return await fn();
       } catch (error) {
         if (i === maxRetries - 1) throw error;
         await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)));
       }
     }
   }
   ```

2. **Logging & Monitoring**  
   - Log errors to Sentry/Datadog.  
   - Track failed transactions in a DB for recovery.

3. **User Feedback**  
   - Display human-readable messages (e.g., "Insufficient funds" instead of raw `CALL_EXCEPTION`).

---

## **Final Example: Robust Ethers Workflow**
```javascript
async function robustEthersOperation() {
  try {
    // 1. Get fallback-supported provider
    const provider = await getProvider();

    // 2. Load wallet safely
    const wallet = loadWalletSafely("PRIVATE_KEY");

    // 3. Estimate gas with retries
    const gasEstimate = await withRetry(() =>
      provider.estimateGas({ to: "0x...", value: ethers.parseEther("0.1") })
    );

    // 4. Send tx with error handling
    const tx = await wallet.sendTransaction({
      to: "0x...",
      value: ethers.parseEther("0.1"),
      gasLimit: gasEstimate,
    });

    console.log("Tx sent:", tx.hash);
  } catch (error) {
    console.error("Critical failure:", error.message);
    // Alert user or retry
  }
}
```

---

### **Summary**
- Always use `try-catch` for `ethers.js` operations.  
- Handle **common errors** (RPC failures, reverts, gas issues).  
- Implement **retries** for transient failures.  
- **Validate inputs** before sending transactions.  
---
# **Exploring Advanced `ethers.js` Features: Multicall, Batch Transactions & Custom Providers**

`ethers.js` supports powerful features for optimizing blockchain interactions, including **multicall** (aggregating contract calls), **batch transactions** (sending multiple txs at once), and **custom providers** (tailored RPC configurations). Below are practical implementations for each.

---

## **1. Multicall: Aggregating Multiple Contract Calls**
**Use Case:** Fetch data from multiple contracts in a single RPC call to reduce latency and costs.  

### **Option A: Using `multicall3` (Recommended)**
The [Multicall3](https://github.com/mds1/multicall) contract aggregates calls efficiently.

```javascript
import { ethers } from "ethers";

const MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"; // Deployed on most chains

async function multicallExample() {
  const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
  const multicall = new ethers.Contract(
    MULTICALL3_ADDRESS,
    [
      "function aggregate(tuple(address target, bytes callData)[] calls) payable returns (tuple(uint256 blockNumber, bytes[] returnData) results)",
    ],
    provider
  );

  // Define calls (e.g., ERC-20 balances)
  const calls = [
    {
      target: "0xTokenA", // Replace with real contract
      callData: new ethers.Interface(["function balanceOf(address) view returns (uint256)"]).encodeFunctionData("balanceOf", ["0xYourAddress"]),
    },
    {
      target: "0xTokenB",
      callData: new ethers.Interface(["function totalSupply() view returns (uint256)"]).encodeFunctionData("totalSupply"),
    },
  ];

  // Execute multicall
  const { returnData } = await multicall.aggregate.staticCall(calls);

  // Decode results
  const balances = returnData.map((data, i) => {
    if (data === "0x") return null; // Handle reverts
    return ethers.AbiCoder.defaultAbiCoder().decode(["uint256"], data)[0];
  });

  console.log("TokenA Balance:", balances[0]);
  console.log("TokenB Total Supply:", balances[1]);
}

multicallExample().catch(console.error);
```

### **Option B: Using `ethers.BatchProvider` (Experimental)**
```javascript
const provider = new ethers.JsonRpcBatchProvider("YOUR_RPC_URL");
const [block, balance] = await Promise.all([
  provider.getBlockNumber(),
  provider.getBalance("0x..."),
]);
```

---

## **2. Batch Transactions: Sending Multiple TXs in One Block**
**Use Case:** Submit multiple transactions atomically (e.g., approvals + swaps).  

### **Using `signer.sendTransaction` in Sequence**
```javascript
const provider = new ethers.JsonRpcProvider("YOUR_RPC_URL");
const signer = new ethers.Wallet("PRIVATE_KEY", provider);

async function sendBatchTransactions() {
  const tx1 = await signer.sendTransaction({
    to: "0x...",
    value: ethers.parseEther("0.1"),
  });

  const tx2 = await signer.sendTransaction({
    to: "0x...",
    value: ethers.parseEther("0.2"),
  });

  await Promise.all([tx1.wait(), tx2.wait()]); // Wait for confirmations
  console.log("Batch TXs completed:", tx1.hash, tx2.hash);
}

sendBatchTransactions().catch(console.error);
```

### **Gas Optimization Tip**
Set a manual `nonce` to ensure transactions are mined in order:
```javascript
const nonce = await provider.getTransactionCount(signer.address);
const tx1 = await signer.sendTransaction({ ..., nonce });
const tx2 = await signer.sendTransaction({ ..., nonce: nonce + 1 });
```

---

## **3. Custom Providers: Tailored RPC Configurations**
**Use Case:** Customize providers for specific chains or optimizations.  

### **A. Custom JSON-RPC Provider**
```javascript
const customProvider = new ethers.JsonRpcProvider("https://custom-rpc.com", {
  chainId: 1234, // Custom chain ID
  name: "My Custom Chain",
  ensAddress: "0x...", // Optional: ENS resolver
});
```

### **B. Fallback Provider (Redundancy)**
```javascript
const providers = [
  new ethers.JsonRpcProvider("https://primary-rpc.com"),
  new ethers.JsonRpcProvider("https://fallback-rpc.com"),
];
const fallbackProvider = new ethers.FallbackProvider(providers, 1); // Quorum: 1
```

### **C. WebSocket Provider (Real-Time Updates)**
```javascript
const wsProvider = new ethers.WebSocketProvider("wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY");

wsProvider.on("block", (blockNumber) => {
  console.log("New block:", blockNumber);
});
```

---

## **4. Advanced: Custom Signers (e.g., Hardware Wallets)**
```javascript
import { LedgerSigner } from "@ethersproject/hardware-wallets";

const ledgerSigner = new LedgerSigner(provider, "hid"); // Or "ble"
const tx = await ledgerSigner.sendTransaction({ to: "0x...", value: ethers.parseEther("1.0") });
```

---

## **5. Error Handling for Advanced Features**
### **Multicall Reverts**
```javascript
try {
  const { returnData } = await multicall.aggregate.staticCall(calls);
} catch (error) {
  if (error.data?.returnData) {
    error.data.returnData.forEach((data, i) => {
      if (data === "0x") console.error(`Call ${i} reverted`);
    });
  }
}
```

### **Batch TX Failures**
```javascript
const receipts = await Promise.allSettled([tx1.wait(), tx2.wait()]);
receipts.forEach((result, i) => {
  if (result.status === "rejected") {
    console.error(`TX ${i + 1} failed:`, result.reason);
  }
});
```

---

## **Summary Table**
| Feature                  | Use Case                          | Implementation Example              |
|--------------------------|-----------------------------------|-------------------------------------|
| **Multicall**            | Aggregate contract calls          | `multicall3.aggregate()`            |
| **Batch Transactions**   | Send multiple TXs sequentially    | `signer.sendTransaction()` + `nonce`|
| **Custom Providers**     | Tailored RPC setups               | `JsonRpcProvider`, `FallbackProvider`|
| **WebSocketProvider**    | Real-time event listening         | `provider.on("block", callback)`    |

---

## **Key Takeaways**
1. **Multicall** reduces RPC roundtrips for contract reads.  
2. **Batch TXs** optimize gas usage by grouping actions.  
3. **Custom Providers** enhance reliability (fallbacks) and performance (WebSockets).  
4. **Hardware Signers** integrate Ledger/Trezor securely.  

For production, consider:  
- **Rate limiting** (Alchemy/Infura tiers).  
- **Gas estimation buffers** (e.g., `gasLimit * 1.2`).  
---
# **Unit Testing and Debugging `ethers.js` Scripts Effectively**

Writing reliable tests and debugging `ethers.js` scripts is critical for smart contract interactions. Below is a structured guide covering **unit testing strategies**, **debugging techniques**, and **best practices** using modern tools.

---

## **1. Unit Testing `ethers.js` Scripts**
### **A. Testing Setup**
Use **Jest** (or **Mocha** + **Chai**) with `ethers.js` and **Mock Providers** to isolate tests from live networks.

#### **Install Dependencies**
```bash
npm install --save-dev jest ethers @typechain/ethers-v5 @typechain/hardhat
```

#### **Example Test File (`test/etherUtils.test.js`)**
```javascript
const { ethers } = require("ethers");
const { MockProvider } = require("ethers-waffle"); // Simulates blockchain

describe("ethers.js Script Tests", () => {
  let provider, signer;

  beforeAll(() => {
    provider = new MockProvider(); // Local test provider
    [signer] = provider.getWallets(); // Test wallet
  });

  it("should fetch the latest block", async () => {
    const block = await provider.getBlock("latest");
    expect(block).toHaveProperty("number");
    expect(block.number).toBeGreaterThan(0);
  });

  it("should send a transaction", async () => {
    const tx = await signer.sendTransaction({
      to: "0x000000000000000000000000000000000000dEaD",
      value: ethers.parseEther("0.1"),
    });
    await tx.wait();
    expect(tx.hash).toMatch(/^0x[a-fA-F0-9]{64}$/);
  });
});
```

---

### **B. Testing Smart Contracts**
Use **Hardhat** or **Foundry** for contract testing with `ethers.js`.

#### **Hardhat Example (`test/MyContract.test.js`)**
```javascript
const { ethers } = require("hardhat");

describe("MyContract", () => {
  let contract;

  beforeAll(async () => {
    const MyContract = await ethers.getContractFactory("MyContract");
    contract = await MyContract.deploy();
  });

  it("should return correct data", async () => {
    expect(await contract.myFunction()).toEqual("expectedValue");
  });
});
```

---

## **2. Debugging `ethers.js` Scripts**
### **A. Common Debugging Techniques**
#### **1. Logging Transactions**
```javascript
const tx = await contract.myFunction();
console.log("Tx Details:", {
  hash: tx.hash,
  gasLimit: tx.gasLimit.toString(),
  data: tx.data,
});
await tx.wait(); // Wait for confirmation
```

#### **2. Inspecting Reverts**
```javascript
try {
  await contract.failingFunction();
} catch (error) {
  console.error("Revert Reason:", error.reason); // e.g., "InsufficientBalance"
  console.error("Decoded Error:", error.data?.data?.message);
}
```

#### **3. Gas Estimation Checks**
```javascript
const estimatedGas = await contract.estimateGas.myFunction();
console.log("Estimated Gas:", estimatedGas.toString());
```

---

### **B. Advanced Debugging Tools**
#### **1. Using `debug` Library**
```bash
npm install debug
```
```javascript
const debug = require("debug")("ethers:debug");

async function debugExample() {
  debug("Fetching block...");
  const block = await provider.getBlock("latest");
  debug("Block: %O", block); // Pretty-print object
}
```

#### **2. Hardhat Console Logs**
```javascript
// In Hardhat scripts
console.log("Storage Slot 0:", await ethers.provider.getStorageAt(contractAddress, 0));
```

#### **3. Tenderly Debugger**
```javascript
const tx = await contract.functionCall();
console.log(`Debug TX: https://dashboard.tenderly.co/tx/${tx.hash}`);
```

---

## **3. Best Practices for Testing & Debugging**
### **A. Testing Best Practices**
| Practice                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Isolate Tests**       | Use `MockProvider` to avoid hitting live networks.                          |
| **Test Edge Cases**     | Test reverts, gas limits, and invalid inputs.                               |
| **Snapshot Testing**    | Use `hardhat_snapshot` to reset state between tests.                        |
| **Coverage Reports**    | Generate reports with `solidity-coverage`.                                  |

### **B. Debugging Best Practices**
| Practice                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Verbose Logging**     | Log `tx.data`, `gasUsed`, and `events`.                                    |
| **Error Classification**| Check `error.code` (e.g., `CALL_EXCEPTION`, `INSUFFICIENT_FUNDS`).         |
| **Use Forks**          | Test on forked mainnet (e.g., `hardhat node --fork <ALCHEMY_URL>`).         |
| **Trace Transactions**  | Use `hardhat-tracer` or Etherscan’s debugger.                              |

---

## **4. Example: End-to-End Test + Debug Workflow**
### **Step 1: Write the Script (`scripts/deploy.js`)**
```javascript
const { ethers } = require("hardhat");

async function main() {
  const Contract = await ethers.getContractFactory("MyContract");
  const contract = await Contract.deploy();
  console.log("Deployed to:", contract.address);
}

main().catch((error) => {
  console.error("Deployment Failed:", error.message);
  process.exitCode = 1;
});
```

### **Step 2: Write the Test (`test/deploy.test.js`)**
```javascript
const { expect } = require("chai");

describe("Deployment", () => {
  it("should deploy the contract", async () => {
    const Contract = await ethers.getContractFactory("MyContract");
    const contract = await Contract.deploy();
    expect(await contract.address).to.be.properAddress;
  });
});
```

### **Step 3: Debug a Failing Test**
```bash
# Run with verbose logging
DEBUG=ethers:* npx hardhat test
```

---

## **5. Key Tools Summary**
| Tool/Package           | Purpose                                      |
|------------------------|----------------------------------------------|
| **Jest/Mocha**         | Test runners.                                |
| **ethers-waffle**      | Mock providers for testing.                  |
| **hardhat-tracer**     | Transaction tracing.                         |
| **Tenderly**           | Cloud-based debugger.                        |
| **solidity-coverage**  | Test coverage reports.                       |

---

## **Final Tips**
1. **Automate Testing**: Integrate with GitHub Actions or CI/CD pipelines.  
2. **Debug in Isolation**: Use `hardhat console` for REPL debugging.  
3. **Monitor Gas**: Always check `gasUsed` in tests.  
