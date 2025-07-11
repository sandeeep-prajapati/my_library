### **Fetching Data from Smart Contracts in React (Token Balances, NFT Metadata, etc.)**  

To fetch data from smart contracts in a React dApp, you’ll typically:  
1. **Connect to a provider** (MetaMask, Infura, Alchemy).  
2. **Initialize a contract instance** using its ABI and address.  
3. **Call read-only functions** (`balanceOf`, `tokenURI`, etc.).  

Here’s a step-by-step guide with code examples for common use cases:

---

## **1. Fetching Token Balances (ERC20, ERC721, Native ETH)**  

### **Example: ERC20 Balance**  
```jsx
import { useState, useEffect } from "react";
import { ethers } from "ethers";

// ERC20 ABI (simplified)
const ERC20_ABI = [
  "function balanceOf(address owner) view returns (uint256)",
];

function TokenBalance({ tokenAddress, userAddress }) {
  const [balance, setBalance] = useState("0");

  useEffect(() => {
    const fetchBalance = async () => {
      // Use MetaMask or a fallback provider
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = new ethers.Contract(tokenAddress, ERC20_ABI, provider);
      
      const rawBalance = await contract.balanceOf(userAddress);
      setBalance(ethers.formatUnits(rawBalance, 18)); // Format for 18 decimals
    };

    fetchBalance();
  }, [tokenAddress, userAddress]);

  return <div>Balance: {balance} tokens</div>;
}
```

**Usage:**  
```jsx
<TokenBalance 
  tokenAddress="0xTokenContractAddress" 
  userAddress="0xUserWalletAddress" 
/>
```

---

### **Example: Native ETH Balance**  
```jsx
useEffect(() => {
  const fetchETHBalance = async () => {
    const provider = new ethers.BrowserProvider(window.ethereum);
    const balance = await provider.getBalance(userAddress);
    setBalance(ethers.formatEther(balance)); // Convert wei → ETH
  };
  fetchETHBalance();
}, [userAddress]);
```

---

## **2. Fetching NFT Metadata (ERC721/ERC1155)**  
NFT metadata is often stored:  
- **On-chain** (via `tokenURI()`).  
- **Off-chain** (IPFS, Arweave, or centralized servers).  

### **Step 1: Fetch `tokenURI`**  
```jsx
const ERC721_ABI = [
  "function tokenURI(uint256 tokenId) view returns (string)",
];

const fetchNFTMetadata = async (contractAddress, tokenId) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const contract = new ethers.Contract(contractAddress, ERC721_ABI, provider);
  const tokenURI = await contract.tokenURI(tokenId);
  return tokenURI; // Returns a URL (e.g., "ipfs://Qm...")
};
```

### **Step 2: Resolve Metadata**  
```jsx
const resolveIPFS = (url) => {
  if (url.startsWith("ipfs://")) {
    return `https://ipfs.io/ipfs/${url.replace("ipfs://", "")}`;
  }
  return url;
};

const response = await fetch(resolveIPFS(tokenURI));
const metadata = await response.json(); // { name, image, attributes, ... }
```

---

## **3. Batch Requests (Multicall)**  
For fetching multiple contract states in one RPC call:  

### **Option A: Ethers.js `Promise.all`**  
```jsx
const [balance, totalSupply] = await Promise.all([
  contract.balanceOf(userAddress),
  contract.totalSupply(),
]);
```

### **Option B: Multicall Contracts**  
Use libraries like:  
- [`ethers-multicall`](https://github.com/EthWorks/ethers-multicall)  
- [`wagmi`’s `useContractReads`](https://wagmi.sh/react/hooks/useContractReads)  

---

## **4. Caching & Performance Optimization**  
To avoid unnecessary RPC calls:  

### **React Query (Recommended)**  
```jsx
import { useQuery } from "@tanstack/react-query";

const { data: balance } = useQuery({
  queryKey: ["tokenBalance", tokenAddress, userAddress],
  queryFn: () => fetchBalance(tokenAddress, userAddress),
  staleTime: 60_000, // Cache for 1 minute
});
```

### **SWR**  
```jsx
import useSWR from "swr";

const { data } = useSWR(
  ["tokenBalance", tokenAddress, userAddress],
  () => fetchBalance(tokenAddress, userAddress)
);
```

---

## **5. Error Handling**  
Always handle:  
- **Contract reverts** (e.g., invalid `tokenId`).  
- **Network issues** (fallback providers).  

```jsx
try {
  const balance = await contract.balanceOf(userAddress);
} catch (err) {
  console.error("Failed to fetch balance:", err);
  setError("Could not load balance");
}
```

---

## **6. Full Example: NFT Gallery**  
```jsx
function NFTGallery({ contractAddress, userAddress }) {
  const [nfts, setNFTs] = useState([]);

  useEffect(() => {
    const fetchNFTs = async () => {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = new ethers.Contract(contractAddress, ERC721_ABI, provider);
      
      // Fetch all NFTs owned by the user
      const balance = await contract.balanceOf(userAddress);
      const tokenIds = await Promise.all(
        Array.from({ length: Number(balance) }, (_, i) =>
          contract.tokenOfOwnerByIndex(userAddress, i)
        )
      );

      // Resolve metadata for each NFT
      const metadata = await Promise.all(
        tokenIds.map((id) => fetchNFTMetadata(contractAddress, id))
      );
      setNFTs(metadata);
    };

    fetchNFTs();
  }, [contractAddress, userAddress]);

  return (
    <div>
      {nfts.map((nft) => (
        <div key={nft.tokenId}>
          <img src={resolveIPFS(nft.image)} alt={nft.name} />
          <h3>{nft.name}</h3>
        </div>
      ))}
    </div>
  );
}
```

---

## **Key Takeaways**  
1. **For balances**: Use `balanceOf` (ERC20) or `getBalance` (ETH).  
2. **For NFTs**: Fetch `tokenURI` and resolve metadata.  
3. **Optimize**: Batch calls with `Promise.all` or multicall.  
4. **Cache**: Use React Query/SWR to reduce RPC calls.  

---

