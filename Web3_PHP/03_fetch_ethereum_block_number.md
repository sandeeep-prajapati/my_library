
---

# Fetch and Display the Latest Ethereum Block Number using Web3 PHP

This guide demonstrates how to write a PHP script to fetch and display the latest Ethereum block number using the `web3.php` library.

---

## **1. Install Web3 PHP Library**

First, install the `web3.php` library using Composer. This library allows you to interact with Ethereum nodes via JSON-RPC.

```bash
composer require web3/web3
```

---

## **2. PHP Script to Fetch the Latest Block Number**

Create a PHP script (e.g., `fetch_block_number.php`) to connect to an Ethereum node (e.g., Infura or a local Hardhat node) and fetch the latest block number.

```php
<?php

require 'vendor/autoload.php';

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

// Replace with your Ethereum node URL (e.g., Infura or local Hardhat node)
$nodeUrl = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'; // For Infura
// $nodeUrl = 'http://localhost:8545'; // For local Hardhat node

// Initialize Web3
$web3 = new Web3(new HttpProvider(new HttpRequestManager($nodeUrl)));

// Variable to store the block number
$blockNumber = 0;

// Fetch the latest block number
$web3->eth->blockNumber(function ($err, $result) use (&$blockNumber) {
    if ($err !== null) {
        die("Error fetching block number: " . $err->getMessage());
    }
    $blockNumber = $result->toString();
});

// Display the latest block number
echo "Latest Ethereum Block Number: " . $blockNumber . PHP_EOL;
```

---

## **3. Explanation of the Script**

### **Step 1: Initialize Web3**
- The `Web3` class is initialized with a provider. In this case, we use `HttpProvider` to connect to an Ethereum node via HTTP.
- Replace the `$nodeUrl` with your Ethereum node URL:
  - For **Infura**: Use your Infura project URL (e.g., `https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID`).
  - For **Hardhat**: Use the local node URL (e.g., `http://localhost:8545`).

### **Step 2: Fetch the Block Number**
- The `blockNumber` method is called on the `eth` object to fetch the latest block number.
- The result is returned as a `BigInteger` object, which is converted to a string using `toString()`.

### **Step 3: Display the Block Number**
- The fetched block number is displayed using `echo`.

---

## **4. Running the Script**

1. Save the script as `fetch_block_number.php`.
2. Run the script from the command line:

```bash
php fetch_block_number.php
```

### **Expected Output**
If the connection is successful, you’ll see the latest Ethereum block number:

```
Latest Ethereum Block Number: 17543210
```

---

## **5. Notes**
- **Infura**: If you’re using Infura, make sure to replace `YOUR_INFURA_PROJECT_ID` with your actual Infura project ID.
- **Hardhat**: If you’re using a local Hardhat node, ensure the node is running at `http://localhost:8545`.
- **Error Handling**: The script includes basic error handling. If the connection fails, an error message will be displayed.

---

## **6. Conclusion**

This script demonstrates how to use the `web3.php` library to interact with an Ethereum node and fetch the latest block number. You can extend this script to perform other Ethereum-related tasks, such as fetching transaction details, account balances, or interacting with smart contracts.