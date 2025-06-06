### 🧑‍💻 Steps to Build a Basic NFT Marketplace

---

### 1. **Smart Contract Development (Backend)**

The backbone of an NFT marketplace is the smart contract, which will handle the core functionalities like **minting**, **buying**, **selling**, and **transferring** NFTs.

#### 1.1 **Set Up Your Development Environment**
To develop and deploy smart contracts, we will use **Hardhat** and **OpenZeppelin** libraries.

- Install **Hardhat**:
  ```bash
  mkdir nft-marketplace
  cd nft-marketplace
  npm init -y
  npm install --save-dev hardhat
  npx hardhat
  ```

- Install **OpenZeppelin** contracts for ERC-721 functionality:
  ```bash
  npm install @openzeppelin/contracts
  ```

#### 1.2 **Create the NFT Contract (ERC-721)**

Create a smart contract using **ERC-721** for NFTs.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract NFTMarketplace is ERC721, Ownable {
    uint256 public tokenCounter;
    mapping(uint256 => uint256) public tokenPrices;
    mapping(address => uint256[]) public userNFTs;

    constructor() ERC721("MarketplaceNFT", "MNFT") {
        tokenCounter = 0;
    }

    // Mint a new NFT
    function mint(string memory tokenURI, uint256 price) external onlyOwner {
        uint256 tokenId = tokenCounter;
        _safeMint(msg.sender, tokenId);
        _setTokenURI(tokenId, tokenURI);
        tokenPrices[tokenId] = price;
        userNFTs[msg.sender].push(tokenId);
        tokenCounter++;
    }

    // Buy an NFT
    function buyNFT(uint256 tokenId) external payable {
        uint256 price = tokenPrices[tokenId];
        require(msg.value >= price, "Insufficient funds");

        address owner = ownerOf(tokenId);
        _transfer(owner, msg.sender, tokenId);
        payable(owner).transfer(price);  // Transfer funds to the owner

        // Update the price (optional) or leave it as fixed
        tokenPrices[tokenId] = 0;
    }

    // List an NFT for sale
    function listNFT(uint256 tokenId, uint256 price) external {
        require(ownerOf(tokenId) == msg.sender, "You must own the token");
        tokenPrices[tokenId] = price;
    }

    // Check if a token exists in a wallet
    function getNFTsByOwner(address owner) external view returns (uint256[] memory) {
        return userNFTs[owner];
    }
}
```

- `mint`: Mints a new NFT for the owner with the provided `tokenURI` and price.
- `buyNFT`: Allows users to buy the NFT, transferring funds to the previous owner.
- `listNFT`: Allows the owner to set a price for the NFT and list it for sale.

---

### 2. **Deploy Smart Contract**

Using **Hardhat**, deploy your contract to the Ethereum testnet (e.g., **Rinkeby** or **Goerli**).

1. Create a deployment script:
```js
// scripts/deploy.js
async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    const NFTMarketplace = await ethers.getContractFactory("NFTMarketplace");
    const marketplace = await NFTMarketplace.deploy();
    console.log("Marketplace contract deployed to:", marketplace.address);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
```

2. Deploy to a testnet:
```bash
npx hardhat run scripts/deploy.js --network rinkeby
```

---

### 3. **Frontend Development (React)**

Now, let’s create a **React frontend** for the marketplace that will allow users to mint, list, buy, and sell NFTs.

#### 3.1 **Set Up React App**

1. Create a React app using **Create React App**:
   ```bash
   npx create-react-app nft-marketplace-frontend
   cd nft-marketplace-frontend
   ```

2. Install dependencies like **Ethers.js** to interact with the Ethereum blockchain:
   ```bash
   npm install ethers
   ```

#### 3.2 **Create Web3 Integration (Using Ethers.js)**

In your frontend, use **Ethers.js** to connect to the Ethereum network and interact with the deployed smart contract.

Example: Connecting to MetaMask and interacting with the marketplace contract.

```js
import { useEffect, useState } from "react";
import { ethers } from "ethers";

const contractAddress = "0xYourContractAddress"; // Your contract address
const contractABI = [
  // Include relevant ABI methods here for minting, listing, and buying
];

function App() {
  const [account, setAccount] = useState(null);
  const [contract, setContract] = useState(null);

  // Connect wallet using MetaMask
  const connectWallet = async () => {
    const provider = new ethers.BrowserProvider(window.ethereum);
    const signer = await provider.getSigner();
    setAccount(await signer.getAddress());

    const nftContract = new ethers.Contract(contractAddress, contractABI, signer);
    setContract(nftContract);
  };

  // Minting an NFT
  const mintNFT = async (tokenURI, price) => {
    await contract.mint(tokenURI, ethers.utils.parseEther(price));
  };

  // Listing an NFT for sale
  const listNFT = async (tokenId, price) => {
    await contract.listNFT(tokenId, ethers.utils.parseEther(price));
  };

  // Buying an NFT
  const buyNFT = async (tokenId) => {
    await contract.buyNFT(tokenId, { value: ethers.utils.parseEther("0.05") });
  };

  useEffect(() => {
    if (window.ethereum) {
      connectWallet();
    }
  }, []);

  return (
    <div>
      <h1>NFT Marketplace</h1>
      {!account && <button onClick={connectWallet}>Connect Wallet</button>}
      {account && <p>Connected: {account}</p>}

      {/* Call mintNFT, listNFT, and buyNFT from the UI */}
    </div>
  );
}

export default App;
```

---

### 4. **IPFS for Metadata Storage**

NFTs need metadata (e.g., images, descriptions), which is typically stored off-chain. We use **IPFS (InterPlanetary File System)** to store the metadata in a decentralized way.

#### Steps:
1. Upload images or metadata files to IPFS using a service like **Pinata** or **Infura**.
2. Store the IPFS URL in the `tokenURI` when minting the NFT.
3. Example `tokenURI`: `ipfs://QmExampleHash`

---

### 5. **Additional Features**

- **User Profiles**: Display the user’s NFTs and transaction history.
- **Search and Filters**: Allow users to search NFTs based on categories, prices, etc.
- **Auctions**: Implement an auction feature where users can bid on NFTs.
- **Royalties**: Implement **ERC-2981** for automatic royalty payments to creators.

---

### 6. **Deploy and Test**

After building the frontend and backend, deploy the entire app to a hosting platform like **Vercel** or **Netlify**. Test the marketplace by connecting your wallet, minting NFTs, listing them for sale, and buying NFTs from others.

---

### 💡 Conclusion

To build a basic NFT marketplace:
1. Develop smart contracts (minting, buying, selling).
2. Deploy contracts to Ethereum testnet.
3. Build a React frontend to interact with the contracts using **Ethers.js**.
4. Store metadata on **IPFS** and link it with your NFTs.
