### **Sending Transactions in React dApps (Minting, Swapping, etc.)**  
To send transactions (e.g., minting NFTs, token swaps) and handle gas fees in a React dApp, follow these steps:

---

## **1. Key Concepts**  
- **Transactions** require:  
  - A **signer** (wallet with ETH for gas).  
  - **Gas fees** (paid in ETH or the network’s native token).  
- **Common Use Cases**:  
  - Minting NFTs.  
  - Token swaps (e.g., Uniswap).  
  - Token transfers (ERC20, ERC721).  

---

## **2. Sending a Transaction (Step-by-Step)**  
### **Step 1: Initialize the Contract with a Signer**  
```javascript
import { ethers } from "ethers";

// Connect to MetaMask
const provider = new ethers.BrowserProvider(window.ethereum);
const signer = await provider.getSigner();

// Initialize contract (replace with your ABI and address)
const contract = new ethers.Contract(
  "0xContractAddress",
  ["function mint() public payable"], // ABI
  signer
);
```

### **Step 2: Send the Transaction**  
```javascript
try {
  // For payable functions (e.g., minting with ETH cost)
  const tx = await contract.mint({
    value: ethers.parseEther("0.01"), // 0.01 ETH
    gasLimit: 300000, // Optional: set gas limit
  });

  console.log("Transaction hash:", tx.hash);

  // Wait for confirmation (usually 1-2 blocks)
  const receipt = await tx.wait();
  console.log("Confirmed in block:", receipt.blockNumber);

} catch (error) {
  console.error("Transaction failed:", error.message);
}
```

---

## **3. Handling Gas Fees**  
### **Option 1: Let MetaMask Estimate Gas**  
MetaMask automatically estimates:  
- **Gas Limit**: Defaults to a safe value.  
- **Gas Price**: Uses the network’s current rate.  

```javascript
const tx = await contract.mint();
```

### **Option 2: Manually Set Gas**  
Override defaults for optimization:  
```javascript
const tx = await contract.mint({
  gasLimit: 250000, // Prevents "out of gas" errors
  maxFeePerGas: ethers.parseUnits("30", "gwei"), // EIP-1559
  maxPriorityFeePerGas: ethers.parseUnits("2", "gwei"),
});
```

### **Option 3: Fetch Current Gas Prices**  
```javascript
const feeData = await provider.getFeeData();
const tx = await contract.mint({
  maxFeePerGas: feeData.maxFeePerGas,
  maxPriorityFeePerGas: feeData.maxPriorityFeePerGas,
});
```

---

## **4. Common Transaction Types**  
### **A. Minting an NFT (ERC721)**  
```javascript
const tx = await contract.mintNFT("0xUserAddress", {
  value: ethers.parseEther("0.05"), // Minting fee
});
```

### **B. Swapping Tokens (Uniswap)**  
```javascript
const routerAddress = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"; // Uniswap Router
const routerABI = [...]; // Uniswap ABI

const router = new ethers.Contract(routerAddress, routerABI, signer);

const tx = await router.swapExactETHForTokens(
  0, // Minimum tokens expected
  ["0xWETH", "0xToken"], // Path (ETH → Token)
  "0xUserAddress",
  Date.now() + 1000 * 60 * 10, // Deadline (10 mins)
  { value: ethers.parseEther("0.1") } // Swap 0.1 ETH
);
```

### **C. Transferring Tokens (ERC20)**  
```javascript
const tokenContract = new ethers.Contract(
  "0xTokenAddress",
  ["function transfer(address to, uint amount) returns (bool)"],
  signer
);

const tx = await tokenContract.transfer(
  "0xRecipientAddress",
  ethers.parseUnits("100", 18) // 100 tokens (18 decimals)
);
```

---

## **5. UX Best Practices**  
### **1. Show Transaction Status**  
```jsx
const [status, setStatus] = useState("");

const mintNFT = async () => {
  setStatus("Awaiting wallet approval...");
  const tx = await contract.mint();
  setStatus("Mining... (Tx hash: " + tx.hash + ")");
  await tx.wait();
  setStatus("Minted successfully!");
};
```

### **2. Handle User Rejections**  
```javascript
try {
  await contract.mint();
} catch (error) {
  if (error.code === "ACTION_REJECTED") {
    alert("User denied transaction");
  } else {
    console.error(error);
  }
}
```

### **3. Estimate Gas Before Sending**  
```javascript
try {
  const estimatedGas = await contract.mint.estimateGas();
  console.log("Estimated gas:", estimatedGas.toString());
} catch (error) {
  console.error("Estimation failed:", error.reason);
}
```

---

## **6. Advanced: Gas Optimization**  
- **Use EIP-1559 Fees**: Prioritize `maxPriorityFeePerGas`.  
- **Batch Transactions**: Combine actions (e.g., mint + approve in one tx).  
- **Gas Tokens**: Use networks with lower fees (Polygon, Arbitrum).  

---

## **7. Full Example: NFT Minting Component**  
```jsx
function MintButton() {
  const [status, setStatus] = useState("");
  const { signer } = useWeb3(); // Custom hook (see previous guide)

  const mint = async () => {
    if (!signer) return alert("Connect wallet first");

    const contract = new ethers.Contract(
      "0xNFTContractAddress",
      ["function mint() public payable"],
      signer
    );

    try {
      setStatus("Awaiting approval...");
      const tx = await contract.mint({ 
        value: ethers.parseEther("0.01"),
      });
      
      setStatus(`Minting... (Tx: ${tx.hash})`);
      await tx.wait();
      setStatus("Success!");

    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  return (
    <div>
      <button onClick={mint}>Mint NFT (0.01 ETH)</button>
      <p>{status}</p>
    </div>
  );
}
```

---

## **Key Takeaways**  
1. **Always use a signer** (wallet connection).  
2. **Handle gas fees** explicitly or let MetaMask estimate.  
3. **Provide user feedback** (pending, success, error states).  
4. **Optimize gas** for cheaper transactions.  
