### **How to Set Up a New Hardhat Project from Scratch**  

This guide walks you through initializing a Hardhat project, installing dependencies, and understanding key configuration files.  

---

## **Step 1: Prerequisites**  
Before starting, ensure you have:  
- **Node.js (v16+ recommended)**  
- **npm or yarn**  
- **A code editor (VS Code recommended)**  

Check Node.js installation:  
```sh
node --version
npm --version
```

---

## **Step 2: Initialize a Hardhat Project**  

### **Option A: Quick Setup (Recommended for Beginners)**  
Run:  
```sh
mkdir my-hardhat-project
cd my-hardhat-project
npm init -y
npm install --save-dev hardhat
npx hardhat init
```
- Choose **"Create a JavaScript project"** (or TypeScript if preferred).  
- Accept defaults for other options.  

### **Option B: Manual Setup (Advanced Users)**  
Install Hardhat manually:  
```sh
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
```
Then create a basic `hardhat.config.js` (more on this below).  

---

## **Step 3: Key Configuration Files**  

After initialization, your project structure will look like this:  
```
my-hardhat-project/
├── contracts/          # Solidity smart contracts
├── scripts/            # Deployment & interaction scripts
├── test/               # Test files (Mocha/Chai/Waffle)
├── hardhat.config.js   # Main Hardhat configuration
└── package.json        # Node.js dependencies
```

### **1. `hardhat.config.js` (Core Configuration)**  
This file defines networks, plugins, and compiler settings.  

#### **Basic Example:**  
```javascript
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.24", // Solidity version
  networks: {
    hardhat: {}, // Local dev network (built-in)
    sepolia: {   // Example: Ethereum testnet
      url: "https://sepolia.infura.io/v3/YOUR_API_KEY",
      accounts: ["0xPRIVATE_KEY"]
    }
  },
  etherscan: {    // Contract verification
    apiKey: "ETHERSCAN_API_KEY"
  }
};
```

#### **Key Configurations:**  
- **`solidity`**: Compiler version (supports multiple versions).  
- **`networks`**: Configure Ethereum networks (Mainnet, Sepolia, etc.).  
- **`plugins`**: Add tools like `hardhat-gas-reporter`.  

---

### **2. `contracts/` (Smart Contracts)**  
- Stores `.sol` files (e.g., `Greeter.sol`).  
- Example:  
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract Greeter {
    string public greeting = "Hello, Hardhat!";
}
```

---

### **3. `scripts/` (Deployment Scripts)**  
- Used for deploying contracts (e.g., `deploy.js`).  
- Example:  
```javascript
const hre = require("hardhat");

async function main() {
  const Greeter = await hre.ethers.getContractFactory("Greeter");
  const greeter = await Greeter.deploy();
  await greeter.waitForDeployment();
  console.log("Greeter deployed to:", await greeter.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Run with:  
```sh
npx hardhat run scripts/deploy.js --network hardhat
```

---

### **4. `test/` (Test Files)**  
- Uses Mocha/Chai for testing (e.g., `greeter-test.js`).  
- Example:  
```javascript
const { expect } = require("chai");

describe("Greeter", function () {
  it("Should return the correct greeting", async function () {
    const Greeter = await ethers.getContractFactory("Greeter");
    const greeter = await Greeter.deploy();
    expect(await greeter.greeting()).to.equal("Hello, Hardhat!");
  });
});
```
Run tests with:  
```sh
npx hardhat test
```

---

## **Step 4: Useful Plugins & Extensions**  
Enhance Hardhat with:  
```sh
npm install --save-dev @nomicfoundation/hardhat-verify hardhat-gas-reporter dotenv
```
- **`hardhat-verify`**: Verify contracts on Etherscan.  
- **`hardhat-gas-reporter`**: Analyze gas costs.  
- **`dotenv`**: Securely store API keys in `.env`.  

---

## **Step 5: Running & Deploying**  
- **Start Hardhat’s local node:**  
  ```sh
  npx hardhat node
  ```
- **Deploy to a testnet (e.g., Sepolia):**  
  ```sh
  npx hardhat run scripts/deploy.js --network sepolia
  ```
- **Verify on Etherscan:**  
  ```sh
  npx hardhat verify --network sepolia DEPLOYED_CONTRACT_ADDRESS
  ```

---

## **Conclusion**  
You now have a fully functional Hardhat project with:  
✅ Smart contracts in `contracts/`  
✅ Deployment scripts in `scripts/`  
✅ Tests in `test/`  
✅ Configurable `hardhat.config.js`  

Next steps:  
- Try deploying to a testnet (e.g., Sepolia).  
- Explore plugins like `hardhat-gas-reporter`.  
- Write more complex contracts & tests.  

Would you like a deep dive into any specific part? 🚀