#### **Topic:** How to send ETH transactions between wallets using `ethers.js`.

Sending ETH transactions between wallets is a fundamental operation in Ethereum development. `ethers.js` simplifies this process by providing a clean and intuitive API for creating, signing, and sending transactions. Below is a step-by-step guide on how to send ETH between wallets using `ethers.js`.

---

### **1. Prerequisites**
- **Node.js**: Ensure Node.js is installed on your machine.
- **ethers.js**: Install `ethers.js` using npm or yarn:
  ```bash
  npm install ethers
  ```
- **Provider**: Use a provider like Infura, Alchemy, or a local Ethereum node to connect to the Ethereum network.
- **Wallet**: You need a wallet with a private key and some ETH (for gas fees).

---

### **2. Setting Up the Provider and Wallet**

First, initialize a provider and a wallet. The wallet will be used to sign and send the transaction.

```javascript
const { ethers } = require("ethers");

// Set up a provider (e.g., Infura)
const INFURA_PROJECT_ID = "YOUR_INFURA_PROJECT_ID";
const provider = new ethers.providers.InfuraProvider("mainnet", INFURA_PROJECT_ID);

// Create a wallet from a private key and connect it to the provider
const privateKey = "YOUR_PRIVATE_KEY"; // Replace with your private key
const wallet = new ethers.Wallet(privateKey, provider);

console.log("Sender Address:", wallet.address);
```

---

### **3. Creating and Sending a Transaction**

To send ETH, you need to create a transaction object and send it using the wallet.

#### **Example: Sending ETH**
```javascript
const { ethers } = require("ethers");

// Set up provider and wallet
const INFURA_PROJECT_ID = "YOUR_INFURA_PROJECT_ID";
const provider = new ethers.providers.InfuraProvider("mainnet", INFURA_PROJECT_ID);
const privateKey = "YOUR_PRIVATE_KEY"; // Replace with your private key
const wallet = new ethers.Wallet(privateKey, provider);

// Define the transaction
const tx = {
  to: "RECIPIENT_ADDRESS", // Replace with the recipient's address
  value: ethers.utils.parseEther("0.01"), // Amount to send (in ETH)
};

// Send the transaction
wallet.sendTransaction(tx).then((transaction) => {
  console.log("Transaction Hash:", transaction.hash);

  // Wait for the transaction to be mined
  return transaction.wait();
}).then((receipt) => {
  console.log("Transaction was mined in block:", receipt.blockNumber);
}).catch((error) => {
  console.error("Error sending transaction:", error);
});
```

---

### **4. Key Components of the Transaction**

- **`to`**: The recipient's Ethereum address.
- **`value`**: The amount of ETH to send, specified in wei. Use `ethers.utils.parseEther` to convert ETH to wei.
- **`gasLimit`**: (Optional) The maximum amount of gas the transaction can use. If not provided, `ethers.js` will estimate it.
- **`gasPrice`**: (Optional) The gas price in wei. If not provided, `ethers.js` will use the current network gas price.

---

### **5. Handling Gas Fees**

Gas fees are required to process transactions on the Ethereum network. `ethers.js` automatically estimates gas limits and gas prices, but you can customize them if needed.

#### **Custom Gas Limit and Gas Price**
```javascript
const tx = {
  to: "RECIPIENT_ADDRESS",
  value: ethers.utils.parseEther("0.01"),
  gasLimit: 21000, // Standard gas limit for ETH transfers
  gasPrice: ethers.utils.parseUnits("50", "gwei"), // Set a custom gas price
};

wallet.sendTransaction(tx).then((transaction) => {
  console.log("Transaction Hash:", transaction.hash);
});
```

---

### **6. Sending Transactions on Testnets**

For testing, you can send transactions on Ethereum testnets like Goerli or Sepolia. Use a testnet provider and ensure your wallet has test ETH (available from faucets).

#### **Example: Sending ETH on Goerli Testnet**
```javascript
const { ethers } = require("ethers");

// Set up a Goerli testnet provider
const INFURA_PROJECT_ID = "YOUR_INFURA_PROJECT_ID";
const provider = new ethers.providers.InfuraProvider("goerli", INFURA_PROJECT_ID);

// Create a wallet from a private key
const privateKey = "YOUR_PRIVATE_KEY"; // Replace with your private key
const wallet = new ethers.Wallet(privateKey, provider);

// Send 0.01 ETH to another address
const tx = {
  to: "RECIPIENT_ADDRESS",
  value: ethers.utils.parseEther("0.01"),
};

wallet.sendTransaction(tx).then((transaction) => {
  console.log("Transaction Hash:", transaction.hash);
});
```

---

### **7. Error Handling**

Always handle errors when sending transactions. Common issues include insufficient balance, incorrect addresses, or network problems.

#### **Example: Error Handling**
```javascript
wallet.sendTransaction(tx).then((transaction) => {
  console.log("Transaction Hash:", transaction.hash);
}).catch((error) => {
  console.error("Error sending transaction:", error);
});
```

---

### **8. Best Practices**
- **Test on Testnets**: Always test your transaction logic on testnets before deploying on mainnet.
- **Secure Private Keys**: Never hardcode private keys in your code. Use environment variables or secure vaults.
- **Gas Optimization**: Use appropriate gas limits and gas prices to avoid overpaying or transaction failures.
- **Confirmations**: Wait for multiple confirmations (e.g., `transaction.wait(3)`) for high-value transactions.

---

By following this guide, you can easily send ETH transactions between wallets using `ethers.js`, enabling you to build powerful and secure Ethereum applications.