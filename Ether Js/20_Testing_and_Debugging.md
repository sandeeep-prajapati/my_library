# **Unit Testing and Debugging `ethers.js` Scripts Effectively**

Writing reliable tests and debugging `ethers.js` scripts is critical for smart contract interactions. Below is a structured guide covering **unit testing strategies**, **debugging techniques**, and **best practices** using modern tools.

---

## **1. Unit Testing `ethers.js` Scripts**
### **A. Testing Setup**
Use **Jest** (or **Mocha** + **Chai**) with `ethers.js` and **Mock Providers** to isolate tests from live networks.

#### **Install Dependencies**
```bash
npm install --save-dev jest ethers @typechain/ethers-v5 @typechain/hardhat
```

#### **Example Test File (`test/etherUtils.test.js`)**
```javascript
const { ethers } = require("ethers");
const { MockProvider } = require("ethers-waffle"); // Simulates blockchain

describe("ethers.js Script Tests", () => {
  let provider, signer;

  beforeAll(() => {
    provider = new MockProvider(); // Local test provider
    [signer] = provider.getWallets(); // Test wallet
  });

  it("should fetch the latest block", async () => {
    const block = await provider.getBlock("latest");
    expect(block).toHaveProperty("number");
    expect(block.number).toBeGreaterThan(0);
  });

  it("should send a transaction", async () => {
    const tx = await signer.sendTransaction({
      to: "0x000000000000000000000000000000000000dEaD",
      value: ethers.parseEther("0.1"),
    });
    await tx.wait();
    expect(tx.hash).toMatch(/^0x[a-fA-F0-9]{64}$/);
  });
});
```

---

### **B. Testing Smart Contracts**
Use **Hardhat** or **Foundry** for contract testing with `ethers.js`.

#### **Hardhat Example (`test/MyContract.test.js`)**
```javascript
const { ethers } = require("hardhat");

describe("MyContract", () => {
  let contract;

  beforeAll(async () => {
    const MyContract = await ethers.getContractFactory("MyContract");
    contract = await MyContract.deploy();
  });

  it("should return correct data", async () => {
    expect(await contract.myFunction()).toEqual("expectedValue");
  });
});
```

---

## **2. Debugging `ethers.js` Scripts**
### **A. Common Debugging Techniques**
#### **1. Logging Transactions**
```javascript
const tx = await contract.myFunction();
console.log("Tx Details:", {
  hash: tx.hash,
  gasLimit: tx.gasLimit.toString(),
  data: tx.data,
});
await tx.wait(); // Wait for confirmation
```

#### **2. Inspecting Reverts**
```javascript
try {
  await contract.failingFunction();
} catch (error) {
  console.error("Revert Reason:", error.reason); // e.g., "InsufficientBalance"
  console.error("Decoded Error:", error.data?.data?.message);
}
```

#### **3. Gas Estimation Checks**
```javascript
const estimatedGas = await contract.estimateGas.myFunction();
console.log("Estimated Gas:", estimatedGas.toString());
```

---

### **B. Advanced Debugging Tools**
#### **1. Using `debug` Library**
```bash
npm install debug
```
```javascript
const debug = require("debug")("ethers:debug");

async function debugExample() {
  debug("Fetching block...");
  const block = await provider.getBlock("latest");
  debug("Block: %O", block); // Pretty-print object
}
```

#### **2. Hardhat Console Logs**
```javascript
// In Hardhat scripts
console.log("Storage Slot 0:", await ethers.provider.getStorageAt(contractAddress, 0));
```

#### **3. Tenderly Debugger**
```javascript
const tx = await contract.functionCall();
console.log(`Debug TX: https://dashboard.tenderly.co/tx/${tx.hash}`);
```

---

## **3. Best Practices for Testing & Debugging**
### **A. Testing Best Practices**
| Practice                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Isolate Tests**       | Use `MockProvider` to avoid hitting live networks.                          |
| **Test Edge Cases**     | Test reverts, gas limits, and invalid inputs.                               |
| **Snapshot Testing**    | Use `hardhat_snapshot` to reset state between tests.                        |
| **Coverage Reports**    | Generate reports with `solidity-coverage`.                                  |

### **B. Debugging Best Practices**
| Practice                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Verbose Logging**     | Log `tx.data`, `gasUsed`, and `events`.                                    |
| **Error Classification**| Check `error.code` (e.g., `CALL_EXCEPTION`, `INSUFFICIENT_FUNDS`).         |
| **Use Forks**          | Test on forked mainnet (e.g., `hardhat node --fork <ALCHEMY_URL>`).         |
| **Trace Transactions**  | Use `hardhat-tracer` or Etherscan’s debugger.                              |

---

## **4. Example: End-to-End Test + Debug Workflow**
### **Step 1: Write the Script (`scripts/deploy.js`)**
```javascript
const { ethers } = require("hardhat");

async function main() {
  const Contract = await ethers.getContractFactory("MyContract");
  const contract = await Contract.deploy();
  console.log("Deployed to:", contract.address);
}

main().catch((error) => {
  console.error("Deployment Failed:", error.message);
  process.exitCode = 1;
});
```

### **Step 2: Write the Test (`test/deploy.test.js`)**
```javascript
const { expect } = require("chai");

describe("Deployment", () => {
  it("should deploy the contract", async () => {
    const Contract = await ethers.getContractFactory("MyContract");
    const contract = await Contract.deploy();
    expect(await contract.address).to.be.properAddress;
  });
});
```

### **Step 3: Debug a Failing Test**
```bash
# Run with verbose logging
DEBUG=ethers:* npx hardhat test
```

---

## **5. Key Tools Summary**
| Tool/Package           | Purpose                                      |
|------------------------|----------------------------------------------|
| **Jest/Mocha**         | Test runners.                                |
| **ethers-waffle**      | Mock providers for testing.                  |
| **hardhat-tracer**     | Transaction tracing.                         |
| **Tenderly**           | Cloud-based debugger.                        |
| **solidity-coverage**  | Test coverage reports.                       |

---

## **Final Tips**
1. **Automate Testing**: Integrate with GitHub Actions or CI/CD pipelines.  
2. **Debug in Isolation**: Use `hardhat console` for REPL debugging.  
3. **Monitor Gas**: Always check `gasUsed` in tests.  
