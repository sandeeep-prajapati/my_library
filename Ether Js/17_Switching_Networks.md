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
