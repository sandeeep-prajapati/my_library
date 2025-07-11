### **Handling Common Web3 Errors in React dApps**  
To build a robust dApp, gracefully handle these frequent Web3 errors:  

| Error Type                | Cause                          | Solution                                                                 |
|---------------------------|--------------------------------|--------------------------------------------------------------------------|
| **Rejected Transaction**  | User denies in MetaMask        | Show a friendly message.                                                 |
| **Wrong Network**         | User is on an unsupported chain | Prompt to switch networks automatically.                                 |
| **Insufficient Gas**      | Gas too low or out of ETH      | Estimate gas properly or suggest topping up.                             |
| **Contract Revert**       | Failed smart contract logic    | Parse revert reason (e.g., "Insufficient balance").                      |
| **No Wallet Installed**   | Missing MetaMask/extension     | Display a wallet installation guide.                                     |

---

## **1. Handling Rejected Transactions**  
When a user clicks "Reject" in MetaMask:  

### **Ethers.js**  
```javascript
try {
  const tx = await contract.mint({ value: ethers.parseEther("0.1") });
} catch (error) {
  if (error.code === "ACTION_REJECTED") {
    alert("You rejected the transaction. Please try again.");
  } else {
    console.error("Transaction failed:", error);
  }
}
```

### **Wagmi (Recommended)**  
```javascript
const { write, error } = useContractWrite({
  address: "0xContract",
  abi: ABI,
  functionName: "mint",
});

useEffect(() => {
  if (error?.name === "UserRejectedRequestError") {
    toast.error("Transaction rejected!");
  }
}, [error]);
```

---

## **2. Handling Wrong Network**  
Automatically prompt users to switch chains:  

### **Ethers.js**  
```javascript
const CHAIN_ID = 1; // Ethereum Mainnet

useEffect(() => {
  const checkNetwork = async () => {
    if (!window.ethereum) return;
    const provider = new ethers.BrowserProvider(window.ethereum);
    const { chainId } = await provider.getNetwork();

    if (chainId !== CHAIN_ID) {
      try {
        await window.ethereum.request({
          method: "wallet_switchEthereumChain",
          params: [{ chainId: `0x${CHAIN_ID.toString(16)}` }],
        });
      } catch (error) {
        if (error.code === 4902) {
          // Chain not added to MetaMask
          await addNetworkToWallet();
        }
      }
    }
  };
  checkNetwork();
}, []);
```

### **Wagmi**  
```javascript
import { useSwitchNetwork, useNetwork } from "wagmi";

function NetworkGuard() {
  const { chain } = useNetwork();
  const { switchNetwork } = useSwitchNetwork();

  useEffect(() => {
    if (chain?.unsupported) {
      switchNetwork?.(1); // Switch to Ethereum
    }
  }, [chain]);
}
```

---

## **3. Handling Insufficient Gas**  
### **Estimate Gas Before Sending**  
```javascript
try {
  const estimatedGas = await contract.mint.estimateGas();
  const tx = await contract.mint({ gasLimit: estimatedGas * 2 }); // Buffer
} catch (error) {
  if (error.message.includes("insufficient funds")) {
    alert("Not enough ETH for gas. Deposit funds and try again.");
  }
}
```

---

## **4. Parsing Contract Reverts**  
Extract human-readable reasons:  

### **Ethers.js v6**  
```javascript
try {
  await contract.transfer("0xInvalidAddress", 100);
} catch (error) {
  const revertReason = error.reason || "Transaction failed";
  alert(`Error: ${revertReason}`); // e.g., "ERC20: transfer to zero address"
}
```

### **Wagmi**  
```javascript
const { error } = useContractWrite({
  address: "0xToken",
  abi: ERC20_ABI,
  functionName: "transfer",
});

console.log(error?.shortMessage); // "execution reverted: ERC20: insufficient balance"
```

---

## **5. No Wallet Installed**  
Guide users to install MetaMask:  

```javascript
if (!window.ethereum) {
  return (
    <div>
      <h3>Install MetaMask to continue</h3>
      <a href="https://metamask.io/download.html" target="_blank">
        Download MetaMask
      </a>
    </div>
  );
}
```

---

## **6. Full Error Handling Component**  
```javascript
function MintButton() {
  const [error, setError] = useState("");

  const mint = async () => {
    try {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const signer = await provider.getSigner();
      const contract = new ethers.Contract(ADDRESS, ABI, signer);

      const tx = await contract.mint({ 
        value: ethers.parseEther("0.1"),
      });
      await tx.wait();
    } catch (err) {
      if (err.code === "ACTION_REJECTED") {
        setError("You rejected the transaction.");
      } else if (err.message.includes("insufficient funds")) {
        setError("Not enough ETH for gas.");
      } else if (err.reason) {
        setError(`Contract error: ${err.reason}`);
      } else {
        setError("Something went wrong.");
      }
    }
  };

  return (
    <div>
      <button onClick={mint}>Mint NFT</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
```

---

## **Key Takeaways**  
1. **Rejected Transactions**: Check for `ACTION_REJECTED` or `UserRejectedRequestError`.  
2. **Wrong Network**: Use `wallet_switchEthereumChain` or Wagmi’s `useSwitchNetwork`.  
3. **Gas Issues**: Estimate gas and add a buffer.  
4. **Contract Reverts**: Parse `error.reason` for human-readable messages.  
5. **No Wallet**: Fallback to educational UI.  

---
