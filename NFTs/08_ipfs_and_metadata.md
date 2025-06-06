## 🧠 What is NFT Metadata?

Metadata describes the NFT—like its name, image, description, and traits.

### Example JSON Metadata (`1.json`)
```json
{
  "name": "Sandeep's NFT",
  "description": "This is a unique digital collectible",
  "image": "ipfs://QmImageHashHere",
  "attributes": [
    { "trait_type": "Rarity", "value": "Legendary" },
    { "trait_type": "Power", "value": "88" }
  ]
}
```

---

## 📁 1. Store Image on IPFS

You can use:

- [Pinata](https://www.pinata.cloud/)
- [NFT.storage](https://nft.storage/)
- [web3.storage](https://web3.storage/)

### Using Pinata (CLI or web UI):

#### a) Upload Image
```bash
pinata pin ./images/nft1.png
```
Returns a CID like: `QmImageHashHere`

---

## 📄 2. Create Metadata JSON

Update `image` to include the IPFS image hash:

```json
{
  "name": "Sandeep's Genesis NFT",
  "description": "A powerful token from Gorakhpur!",
  "image": "ipfs://QmImageHashHere",
  "attributes": [
    { "trait_type": "Power Level", "value": 99 }
  ]
}
```

---

## 📤 3. Upload Metadata JSON to IPFS

```bash
pinata pin ./metadata/1.json
```
Returns: `QmMetaHashHere`

---

## 🔗 4. Link Metadata in Your Smart Contract

When minting, the `tokenURI` should return:

```solidity
function tokenURI(uint256 tokenId) public view override returns (string memory) {
    return string(abi.encodePacked(baseTokenURI, Strings.toString(tokenId), ".json"));
}
```

If your base URI is:

```solidity
constructor() ERC721("MyNFT", "MNFT") {
    baseTokenURI = "ipfs://QmMetaHashHere/";
}
```

Then `tokenURI(1)` will return:
```
ipfs://QmMetaHashHere/1.json
```

> ✅ Marketplaces like OpenSea will automatically fetch and display this metadata!

---

## 🛡 Best Practices for NFT Metadata on IPFS

| ✅ Good Practice | 💡 Why It Matters |
|------------------|-------------------|
| Pin files on **Pinata**/**NFT.storage** | Ensures files stay on IPFS and don’t get garbage-collected |
| Use **immutable** metadata | Don’t change after minting unless you're doing "dynamic NFTs" |
| Format with **ERC-721 metadata standard** | Ensures OpenSea and wallets can read it |
| Include `attributes` as traits | Enables rarity filters and better UX in marketplaces |
| Store **both image & metadata on IPFS** | Full decentralization—no central server dependency |
| Upload all metadata **before minting** | TokenURI must point to a real file on-chain from day 1 |

---

## 🔥 Bonus: Automate with `pinata-sdk`

You can automate uploads using Node.js:
```bash
npm install @pinata/sdk
```

```js
const pinataSDK = require('@pinata/sdk');
const pinata = pinataSDK('YOUR_API_KEY', 'YOUR_SECRET_KEY');

const readableStreamForFile = fs.createReadStream('./images/nft1.png');
pinata.pinFileToIPFS(readableStreamForFile).then((result) => {
  console.log(result.IpfsHash);
});
```

---

Let me know if you'd like a ready-made script that:
- Uploads images
- Generates metadata
- Uploads metadata to IPFS
- Outputs tokenURIs for minting 🚀
