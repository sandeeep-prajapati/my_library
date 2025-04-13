## 🧱 Prerequisites
You’ll need:
- **Solidity >=0.8.0**
- **OpenZeppelin contracts** for safety & simplicity
- Tools like **Hardhat** or **Remix** to deploy and test

---

## 📝 Basic NFT Smart Contract (ERC-721)

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

---

## 🔍 Breakdown of Key Parts

### ✅ 1. **Inherits ERC721**
```solidity
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
```
Gives you access to built-in NFT logic.

---

### ✅ 2. **Unique Token IDs**
```solidity
uint256 public nextTokenId;
```
Auto-increments with each mint.

---

### ✅ 3. **Mint Function**
```solidity
function mint(address to) public onlyOwner {
    _safeMint(to, nextTokenId);
    nextTokenId++;
}
```
- Only the **contract owner** can mint
- `_safeMint` handles checks to avoid sending to invalid addresses

---

### ✅ 4. **Token URI Metadata**
```solidity
function _baseURI() internal view override returns (string memory) {
    return baseTokenURI;
}
```
The final `tokenURI` will be:
```
baseTokenURI + tokenId
e.g., https://mynft.com/metadata/1
```

---

## 🔧 Core ERC-721 Functions (Inherited)

| Function | Purpose |
|----------|---------|
| `balanceOf(address)` | Get total NFTs owned by an address |
| `ownerOf(tokenId)` | Who owns a specific NFT |
| `transferFrom(from, to, tokenId)` | Transfer ownership |
| `approve(to, tokenId)` | Approve someone to transfer your NFT |
| `getApproved(tokenId)` | Check approved account |
| `tokenURI(tokenId)` | Get metadata (image, name, etc.) |

---

## 🧪 Bonus: Example Metadata JSON (off-chain)

```json
{
  "name": "My Cool NFT",
  "description": "A unique digital asset!",
  "image": "ipfs://Qm123abc456...",
  "attributes": [
    { "trait_type": "Background", "value": "Blue" },
    { "trait_type": "Eyes", "value": "Laser" }
  ]
}
```
