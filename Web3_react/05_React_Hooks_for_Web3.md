### **Creating Custom React Hooks for Web3 (`useWeb3`, `useContract`)**  
Custom hooks encapsulate Web3 logic for reusability across components. Below are clean, TypeScript-friendly implementations for:  
1. **`useWeb3`** – Manage wallet connections and provider state.  
2. **`useContract`** – Simplify contract interactions.  

---

## **1. `useWeb3` Hook (Wallet/Provider State)**  
Handles MetaMask connection, account changes, and network switches.  

### **Code (`useWeb3.ts`)**
```typescript
import { useState, useEffect } from "react";
import { ethers } from "ethers";

export function useWeb3() {
  const [provider, setProvider] = useState<ethers.BrowserProvider | null>(null);
  const [signer, setSigner] = useState<ethers.JsonRpcSigner | null>(null);
  const [address, setAddress] = useState<string>("");
  const [chainId, setChainId] = useState<number>(0);

  // Initialize provider and listen for changes
  useEffect(() => {
    if (!window.ethereum) return;

    const initProvider = async () => {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const network = await provider.getNetwork();
      
      setProvider(provider);
      setChainId(Number(network.chainId));

      // Fetch signer if wallet is connected
      const accounts = await window.ethereum.request({ method: "eth_accounts" });
      if (accounts.length > 0) {
        const signer = await provider.getSigner();
        setSigner(signer);
        setAddress(await signer.getAddress());
      }
    };

    initProvider();

    // Event listeners
    const handleAccountsChanged = (accounts: string[]) => {
      if (accounts.length === 0) {
        setSigner(null);
        setAddress("");
      } else {
        provider?.getSigner().then((s) => {
          setSigner(s);
          s.getAddress().then(setAddress);
        });
      }
    };

    const handleChainChanged = (chainId: string) => {
      setChainId(Number(chainId));
      window.location.reload(); // Prevent state corruption
    };

    window.ethereum.on("accountsChanged", handleAccountsChanged);
    window.ethereum.on("chainChanged", handleChainChanged);

    return () => {
      window.ethereum.removeListener("accountsChanged", handleAccountsChanged);
      window.ethereum.removeListener("chainChanged", handleChainChanged);
    };
  }, []);

  // Connect wallet
  const connect = async () => {
    if (!window.ethereum) throw new Error("MetaMask not installed");
    await window.ethereum.request({ method: "eth_requestAccounts" });
    const signer = await provider!.getSigner();
    setSigner(signer);
    setAddress(await signer.getAddress());
  };

  return { provider, signer, address, chainId, connect };
}
```

### **Usage in Components**
```tsx
function WalletButton() {
  const { address, connect } = useWeb3();

  return (
    <button onClick={connect}>
      {address ? `${address.slice(0, 6)}...` : "Connect Wallet"}
    </button>
  );
}
```

---

## **2. `useContract` Hook (Contract Interactions)**  
Simplifies calling contract functions with built-in error handling.  

### **Code (`useContract.ts`)**
```typescript
import { useState, useEffect } from "react";
import { ethers } from "ethers";

export function useContract<T extends ethers.Contract>(
  address: string,
  abi: ethers.InterfaceAbi,
  signer?: ethers.Signer | null
) {
  const [contract, setContract] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize contract when signer changes
  useEffect(() => {
    if (!address || !abi || !signer) return;
    const contract = new ethers.Contract(address, abi, signer) as T;
    setContract(contract);
  }, [address, abi, signer]);

  // Wrapper for contract calls
  const call = async <K extends keyof T>(
    method: K,
    ...args: Parameters<T[K]>
  ): Promise<ReturnType<T[K]> | undefined> => {
    if (!contract) throw new Error("Contract not initialized");
    setLoading(true);
    setError(null);

    try {
      const tx = await contract[method](...args);
      const result = await tx.wait(); // For transactions
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return { contract, call, loading, error };
}
```

### **Usage Example (ERC20 Token)**
```tsx
function TransferTokens() {
  const { signer } = useWeb3();
  const { contract, call, loading, error } = useContract<ethers.Contract>(
    "0xTokenAddress",
    ["function transfer(address to, uint256 amount) returns (bool)"],
    signer
  );

  const handleTransfer = async () => {
    const result = await call("transfer", "0xRecipient", ethers.parseEther("1"));
    if (result) console.log("Transfer succeeded!");
  };

  return (
    <div>
      <button onClick={handleTransfer} disabled={loading}>
        {loading ? "Sending..." : "Send 1 Token"}
      </button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
```

---

## **3. Key Benefits of Custom Hooks**  
1. **Reusability**  
   - Use the same logic across components (e.g., `useWeb3` in navbar + contract pages).  
2. **Separation of Concerns**  
   - Isolate Web3 logic from UI components.  
3. **Type Safety**  
   - TypeScript support for contracts and methods.  
4. **Error Handling**  
   - Centralized error states (`loading`, `error`).  

---

## **4. Advanced Optimizations**  
### **Memoization (Prevent Unnecessary Renders)**  
```typescript
import { useMemo } from "react";

const { contract } = useContract(address, abi, signer);
const memoizedContract = useMemo(() => contract, [contract]);
```

### **Batch Calls (Multicall)**  
```typescript
const calls = [
  contract.balanceOf(address),
  contract.totalSupply()
];
const [balance, supply] = await Promise.all(calls);
```

---

## **5. Alternatives (Wagmi/viem)**  
For complex dApps, consider:  
- **[Wagmi](https://wagmi.sh/)** – Pre-built hooks like `useAccount`, `useContractRead`.  
- **[viem](https://viem.sh/)** – Lightweight alternative to Ethers.js.  

**Example with Wagmi:**  
```tsx
import { useContractRead } from "wagmi";

const { data: balance } = useContractRead({
  address: "0xToken",
  abi: erc20ABI,
  functionName: "balanceOf",
  args: ["0xUser"],
});
```

---

## **Conclusion**  
1. **`useWeb3`** – Manages wallet connections and network states.  
2. **`useContract`** – Simplifies contract calls with error handling.  
3. **Combine them** for a clean, maintainable dApp architecture.  

---
