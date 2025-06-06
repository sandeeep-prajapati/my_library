Let’s explore how to **implement royalties**, especially using the **ERC-2981 standard**, which is widely supported by marketplaces like OpenSea, LooksRare, and more.

---

## 🔧 What is ERC-2981?

ERC-2981 is a royalty standard for NFTs that defines how to signal **royalty payment information** to marketplaces and platforms.

### Key Features:
- Royalties are off-chain enforced (via marketplace support).
- Royalty info includes: **receiver address** and **percentage (in basis points)**.

---

## 📦 Step 1: Install OpenZeppelin Contracts

```bash
npm install @openzeppelin/contracts
```

---

## 🧱 Step 2: Inherit `ERC721`, `ERC2981`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/common/ERC2981.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyRoyaltyNFT is ERC721, ERC2981, Ownable {
    uint256 public tokenCounter;

    constructor() ERC721("RoyaltyNFT", "RNFT") {
        tokenCounter = 0;

        // Set default royalty to 5% (500 basis points)
        _setDefaultRoyalty(msg.sender, 500);
    }

    function mint(address to) external onlyOwner {
        _safeMint(to, tokenCounter);
        tokenCounter++;
    }

    // Optional: Set royalty per token
    function setTokenRoyalty(
        uint256 tokenId,
        address receiver,
        uint96 feeNumerator
    ) external onlyOwner {
        _setTokenRoyalty(tokenId, receiver, feeNumerator);
    }

    // Overrides required by Solidity
    function supportsInterface(bytes4 interfaceId)
        public
        view
        virtual
        override(ERC721, ERC2981)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
```

---

## 📘 Royalty Explained

- **500 basis points = 5%**
- You can call:
  ```solidity
  _setDefaultRoyalty(address receiver, uint96 feeNumerator);
  ```
- Or set per token:
  ```solidity
  _setTokenRoyalty(tokenId, receiver, feeNumerator);
  ```

> 🧠 Royalty payment logic is **not enforced on-chain**, it's enforced by **marketplaces that support ERC-2981**.

---

## 🔎 Marketplaces like OpenSea

They read royalty info like this:

```solidity
function royaltyInfo(uint256 _tokenId, uint256 _salePrice)
    external
    view
    returns (address receiver, uint256 royaltyAmount);
```

So if someone sells your NFT for 1 ETH, and royalty is 5%, `royaltyInfo()` returns:
- Receiver address: your wallet
- Royalty amount: `0.05 ETH`

---

## ✅ Best Practices

| Tip | Why |
|-----|-----|
| Use `ERC2981` instead of custom logic | Ensures cross-platform royalty enforcement |
| Use basis points (10000 = 100%) | Prevents floating-point errors |
| Use `_setDefaultRoyalty()` for global config | Simple for collections with uniform royalties |
| Override `supportsInterface()` | So OpenSea and others recognize the royalty interface |
