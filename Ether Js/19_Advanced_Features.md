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
