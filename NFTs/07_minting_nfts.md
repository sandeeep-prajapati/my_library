## 🔄 What is Minting?

**Minting** is the process of creating a new NFT by assigning a unique `tokenId` and giving it to a wallet address. This records ownership on the blockchain.

---

## 🧱 1. Solidity: Implementing `mint` Function

In your ERC-721 smart contract (`MyNFT.sol`), we define a `mint()` function:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyNFT is ERC721, Ownable {
    uint256 public nextTokenId;
    string public baseTokenURI;

    constructor(string memory _baseTokenURI) ERC721("MyNFT", "MNFT") {
        baseTokenURI = _baseTokenURI;
    }

    function mint(address to) public onlyOwner {
        _safeMint(to, nextTokenId);
        nextTokenId++;
    }

    function _baseURI() internal view override returns (string memory) {
        return baseTokenURI;
    }
}
```

### 🔑 Key Details:
- `onlyOwner`: restricts minting to the contract owner.
- `nextTokenId`: auto-increments to ensure uniqueness.
- `_safeMint`: ensures the address can accept NFTs.

---

## 🛠️ 2. Hardhat: Deploy and Mint via Script

### ✅ a) Deploy the contract (`scripts/deploy.js`):

```javascript
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  const MyNFT = await hre.ethers.getContractFactory("MyNFT");
  const nft = await MyNFT.deploy("https://mynft.example/api/");
  await nft.deployed();

  console.log("MyNFT deployed to:", nft.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

### ✅ b) Mint a token (`scripts/mint.js`):

```javascript
const hre = require("hardhat");

async function main() {
  const [owner] = await hre.ethers.getSigners();
  const contractAddress = "0xYourDeployedContractAddress";

  const MyNFT = await hre.ethers.getContractAt("MyNFT", contractAddress);

  const tx = await MyNFT.mint(owner.address);
  await tx.wait();

  console.log("NFT minted to:", owner.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

---

## 🧪 3. Run Deployment and Mint Scripts

```bash
# Deploy contract
npx hardhat run scripts/deploy.js --network sepolia

# Mint NFT to your wallet
npx hardhat run scripts/mint.js --network sepolia
```

---

## 🧰 4. Optional: Mint from Console

```bash
npx hardhat console --network sepolia
```

Then run:

```js
const nft = await ethers.getContractAt("MyNFT", "0xYourContractAddress");
await nft.mint("0xRecipientAddress");
```

---

## 🔍 5. View on Testnet Explorer

Once minted:
- Check `ownerOf(tokenId)` returns the recipient's address
- Check metadata using `tokenURI(tokenId)` (if implemented)

---

## 🧠 Summary

| Step | What it Does |
|------|--------------|
| `mint(to)` | Mints a new NFT to the given address |
| `nextTokenId++` | Increments token ID for uniqueness |
| `_baseURI()` | Sets the base for `tokenURI()` |
| Hardhat scripts | Deploy and interact with smart contract |
