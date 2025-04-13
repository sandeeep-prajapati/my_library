## 🧰 Prerequisites

Install Ethers.js:
```bash
npm install ethers
```

And connect to a testnet (e.g., Sepolia) using:
- An **Infura**, **Alchemy**, or **public RPC** endpoint
- A **private key** or **MetaMask signer**

---

## 📝 1. Connect to Ethereum via Ethers.js

```js
const { ethers } = require("ethers");

const provider = new ethers.JsonRpcProvider("https://sepolia.infura.io/v3/YOUR_INFURA_ID");
const privateKey = "0xYOUR_PRIVATE_KEY"; // Keep this safe!
const signer = new ethers.Wallet(privateKey, provider);
```

---

## 📄 2. Define Contract ABI & Address

You need the **ERC-721 ABI** and your **contract address**:

```js
const contractAddress = "0xYourNFTContractAddress";

const abi = [
  "function mint(address to) public",
  "function transferFrom(address from, address to, uint256 tokenId) public",
  "function ownerOf(uint256 tokenId) public view returns (address)",
  "function tokenURI(uint256 tokenId) public view returns (string)"
];

const nft = new ethers.Contract(contractAddress, abi, signer);
```

---

## 🪙 3. Mint an NFT

```js
const to = "0xRecipientAddress";
const tx = await nft.mint(to);
await tx.wait();
console.log(`NFT minted to ${to}`);
```

---

## 🔁 4. Transfer an NFT

```js
const from = await signer.getAddress();  // current owner
const to = "0xNewOwnerAddress";
const tokenId = 0;

const tx = await nft.transferFrom(from, to, tokenId);
await tx.wait();
console.log(`NFT #${tokenId} transferred from ${from} to ${to}`);
```

---

## 🔍 5. Check Owner and Metadata

```js
const owner = await nft.ownerOf(tokenId);
console.log(`Owner of token ${tokenId} is:`, owner);

const uri = await nft.tokenURI(tokenId);
console.log(`Metadata URI:`, uri);
```

---

## 💡 Bonus: Connect via MetaMask in Frontend

```js
const provider = new ethers.BrowserProvider(window.ethereum);
const signer = await provider.getSigner();
const nft = new ethers.Contract(contractAddress, abi, signer);
```

> Then call:
```js
await nft.mint(await signer.getAddress());
```

---

## ✅ Summary

| Function       | Purpose                                 |
|----------------|------------------------------------------|
| `mint(to)`     | Mints a new NFT to a given address       |
| `transferFrom(from, to, id)` | Transfers NFT ownership      |
| `ownerOf(id)`  | Checks current owner of the token        |
| `tokenURI(id)` | Fetches metadata URI for the NFT         |
