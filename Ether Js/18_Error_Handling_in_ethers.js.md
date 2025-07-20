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
