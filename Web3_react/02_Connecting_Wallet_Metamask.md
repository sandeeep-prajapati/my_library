### **Connecting MetaMask to a React App with `window.ethereum`**  
A step-by-step guide to integrating MetaMask, handling account/network changes, and best practices for production dApps.

---

## **1. Basic Wallet Connection**  
### **Step 1: Check for MetaMask**  
Ensure the user has MetaMask installed:  
```javascript
if (!window.ethereum) {
  alert("Please install MetaMask!");
  window.open("https://metamask.io/download.html", "_blank");
}
```

### **Step 2: Connect Wallet (EIP-1193 Standard)**  
```javascript
import { useState, useEffect } from "react";

function App() {
  const [account, setAccount] = useState("");

  const connectWallet = async () => {
    try {
      const accounts = await window.ethereum.request({ 
        method: "eth_requestAccounts" 
      });
      setAccount(accounts[0]);
    } catch (error) {
      console.error("User rejected request:", error);
    }
  };

  return (
    <div>
      {account ? (
        <p>Connected: {account}</p>
      ) : (
        <button onClick={connectWallet}>Connect MetaMask</button>
      )}
    </div>
  );
}
```

---

## **2. Handling Account/Network Changes**  
### **Step 1: Listen for Account Changes**  
```javascript
useEffect(() => {
  const handleAccountsChanged = (accounts) => {
    if (accounts.length === 0) {
      console.log("Disconnected");
      setAccount("");
    } else {
      setAccount(accounts[0]);
    }
  };

  window.ethereum.on("accountsChanged", handleAccountsChanged);
  return () => window.ethereum.removeListener("accountsChanged", handleAccountsChanged);
}, []);
```

### **Step 2: Listen for Network Changes**  
```javascript
useEffect(() => {
  const handleChainChanged = (chainId) => {
    console.log("Switched to chain:", chainId);
    window.location.reload(); // Recommended to avoid state issues
  };

  window.ethereum.on("chainChanged", handleChainChanged);
  return () => window.ethereum.removeListener("chainChanged", handleChainChanged);
}, []);
```

---

## **3. Full Production-Ready Example**  
```javascript
import { useState, useEffect } from "react";

export default function WalletButton() {
  const [account, setAccount] = useState("");
  const [chainId, setChainId] = useState("");

  // Check if wallet is connected on load
  useEffect(() => {
    const checkConnection = async () => {
      if (window.ethereum?.isMetaMask) {
        const accounts = await window.ethereum.request({ method: "eth_accounts" });
        if (accounts.length > 0) setAccount(accounts[0]);
        
        const currentChainId = await window.ethereum.request({ method: "eth_chainId" });
        setChainId(currentChainId);
      }
    };
    checkConnection();
  }, []);

  // Connect wallet
  const connectWallet = async () => {
    try {
      const accounts = await window.ethereum.request({ 
        method: "eth_requestAccounts" 
      });
      setAccount(accounts[0]);
    } catch (error) {
      console.error("User denied access:", error);
    }
  };

  // Handle events
  useEffect(() => {
    if (!window.ethereum) return;

    const handleAccountsChanged = (accounts) => {
      setAccount(accounts[0] || "");
    };

    const handleChainChanged = (chainId) => {
      setChainId(chainId);
      window.location.reload(); // Avoid state mismatches
    };

    window.ethereum.on("accountsChanged", handleAccountsChanged);
    window.ethereum.on("chainChanged", handleChainChanged);

    return () => {
      window.ethereum.removeListener("accountsChanged", handleAccountsChanged);
      window.ethereum.removeListener("chainChanged", handleChainChanged);
    };
  }, []);

  return (
    <div>
      {account ? (
        <div>
          <p>Address: {account.slice(0, 6)}...{account.slice(-4)}</p>
          <p>Chain ID: {chainId}</p>
        </div>
      ) : (
        <button onClick={connectWallet}>Connect MetaMask</button>
      )}
    </div>
  );
}
```

---

## **4. Key Considerations**  
### **1. Error Handling**  
- **User rejects connection:** Catch `eth_requestAccounts` errors.  
- **Wrong network:** Use `chainId` to verify (e.g., `0x1` for Ethereum Mainnet).  

### **2. Security**  
- **Always reload on chain changes** to prevent state corruption.  
- **Never store private keys** in React state.  

### **3. UX Improvements**  
- **Add a "Disconnect" button** (MetaMask doesn’t have true logout, but you can reset state):  
  ```javascript
  const disconnect = () => setAccount("");
  ```
- **Show network name** (e.g., "Ethereum Mainnet" instead of `0x1`).  

---

## **5. Alternatives to `window.ethereum`**  
### **1. Ethers.js**  
```javascript
import { ethers } from "ethers";

const provider = new ethers.BrowserProvider(window.ethereum);
const signer = await provider.getSigner();
console.log("Address:", await signer.getAddress());
```

### **2. Web3.js**  
```javascript
import Web3 from "web3";

const web3 = new Web3(window.ethereum);
const accounts = await web3.eth.getAccounts();
```

### **3. Wagmi (Recommended for Production)**  
```bash
npm install wagmi viem
```
```javascript
import { useAccount, useConnect, useDisconnect } from "wagmi";
import { injected } from "wagmi/connectors";

function WalletButton() {
  const { connect } = useConnect();
  const { disconnect } = useDisconnect();
  const { address, isConnected } = useAccount();

  return (
    <button 
      onClick={() => isConnected ? disconnect() : connect({ connector: injected() })}
    >
      {isConnected ? address : "Connect MetaMask"}
    </button>
  );
}
```

---

## **6. Troubleshooting**  
| **Issue**                  | **Solution**                                      |
|----------------------------|--------------------------------------------------|
| `window.ethereum` undefined | Ensure MetaMask is installed. Use `@metamask/onboarding` as fallback. |
| "User rejected request"     | Handle the error gracefully in UI.               |
| Stale chain/account data    | Always reload on `chainChanged`.                 |

---

## **Conclusion**  
1. Use `window.ethereum.request({ method: "eth_requestAccounts" })` for connection.  
2. Listen to `accountsChanged` and `chainChanged` events.  
3. For production, consider **Wagmi** or **Ethers.js** for cleaner code.  
