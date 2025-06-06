## ✅ Prerequisites

Make sure you have:

1. **Node.js + npm** installed  
2. **Hardhat** project set up  
3. An **Infura/Alchemy** API key  
4. A **MetaMask wallet** with **testnet ETH**

---

## 🛠 Step 1: Initialize a Hardhat Project

```bash
mkdir MyNFT
cd MyNFT
npm init -y
npm install --save-dev hardhat
npx hardhat
```

Choose **"Create a basic sample project"** when prompted.

---

## 📦 Step 2: Install OpenZeppelin Contracts

```bash
npm install @openzeppelin/contracts
```

---

## 📝 Step 3: Write the NFT Contract

Create `contracts/MyNFT.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyNFT is ERC721, Ownable {
    uint256 public tokenId;
    string public baseTokenURI;

    constructor(string memory _baseTokenURI) ERC721("MyNFT", "MNFT") {
        baseTokenURI = _baseTokenURI;
    }

    function mint(address to) public onlyOwner {
        _safeMint(to, tokenId);
        tokenId++;
    }

    function _baseURI() internal view override returns (string memory) {
        return baseTokenURI;
    }
}
```

---

## 📤 Step 4: Create the Deploy Script

Create `scripts/deploy.js`:

```javascript
async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying contract with:", deployer.address);

  const MyNFT = await ethers.getContractFactory("MyNFT");
  const nft = await MyNFT.deploy("https://mynft.example/api/");

  await nft.deployed();

  console.log("NFT deployed to:", nft.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

---

## 🌐 Step 5: Configure Hardhat for Testnet

Update `hardhat.config.js`:

```js
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.20",
  networks: {
    sepolia: {
      url: "https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID",
      accounts: ["YOUR_PRIVATE_KEY"]
    }
  }
};
```

> 🔒 Don't expose your private key! Use `.env` for safety:
```bash
npm install dotenv
```
In `.env`:
```
PRIVATE_KEY=your_wallet_private_key
INFURA_API_KEY=your_infura_key
```
Then update config:
```js
require("dotenv").config();

sepolia: {
  url: `https://sepolia.infura.io/v3/${process.env.INFURA_API_KEY}`,
  accounts: [process.env.PRIVATE_KEY]
}
```

---

## 🚀 Step 6: Deploy to Sepolia (or Goerli)

```bash
npx hardhat run scripts/deploy.js --network sepolia
```

You'll see output like:
```
Deploying contract with: 0xYourWalletAddress
NFT deployed to: 0xContractAddress
```

---

## 🔍 Step 7: Verify the Contract (Optional)

Install plugin:

```bash
npm install --save-dev @nomicfoundation/hardhat-verify
```

In `hardhat.config.js`:
```js
etherscan: {
  apiKey: process.env.ETHERSCAN_API_KEY
}
```

Then run:

```bash
npx hardhat verify --network sepolia 0xContractAddress "https://mynft.example/api/"
```

---

## 🧪 Done! You can now:
- Mint NFTs using `mint()` from owner address
- View your contract on **[Sepolia Etherscan](https://sepolia.etherscan.io/)**
- Later, connect it to a frontend (like with React + Web3.js or Ethers.js)
