Here’s a Markdown file (`08_call_custom_smart_contract.md`) that explains how to write a PHP script to call a "view" function (like `getTotalSupply`) from a custom smart contract using Web3 PHP.

---

# Call a View Function from a Custom Smart Contract using Web3 PHP

This guide demonstrates how to write a PHP script to call a "view" function (e.g., `getTotalSupply`) from a custom smart contract using the `web3.php` library.

---

## **1. Install Web3 PHP Library**

First, install the `web3.php` library using Composer. This library allows you to interact with Ethereum nodes via JSON-RPC.

```bash
composer require web3/web3
```

---

## **2. Write the PHP Script**

Create a PHP script (e.g., `call_contract_function.php`) to call a "view" function from your custom smart contract.

### **Step 1: Define the Contract ABI and Address**
You need the ABI (Application Binary Interface) of your smart contract and its deployed address.

```php
<?php

require 'vendor/autoload.php';

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Web3\Contract;

// Replace with your Ethereum node URL (e.g., Infura or local Hardhat node)
$nodeUrl = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'; // For Infura
// $nodeUrl = 'http://localhost:8545'; // For local Hardhat node

// Initialize Web3
$web3 = new Web3(new HttpProvider(new HttpRequestManager($nodeUrl)));

// Replace with your contract's ABI and address
$contractABI = '[{"constant":true,"inputs":[],"name":"getTotalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}]';
$contractAddress = '0xYourContractAddress';

// Initialize the contract
$contract = new Contract($web3->provider, $contractABI);
$contract->at($contractAddress);

// Variable to store the result
$result = '';

// Call the "getTotalSupply" function
$contract->call('getTotalSupply', function ($err, $response) use (&$result) {
    if ($err !== null) {
        die("Error calling contract function: " . $err->getMessage());
    }
    $result = $response[0]->toString(); // Convert BigNumber to string
});

// Display the result
echo "Total Supply: " . $result . PHP_EOL;
```

---

## **3. Explanation of the Script**

### **Step 1: Initialize Web3**
- The `Web3` class is initialized with a provider. In this case, we use `HttpProvider` to connect to an Ethereum node via HTTP.
- Replace the `$nodeUrl` with your Ethereum node URL:
  - For **Infura**: Use your Infura project URL (e.g., `https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID`).
  - For **Hardhat**: Use the local node URL (e.g., `http://localhost:8545`).

### **Step 2: Define the Contract ABI and Address**
- The `$contractABI` variable contains the ABI of your smart contract. Replace it with the actual ABI of your contract.
- The `$contractAddress` variable contains the deployed address of your smart contract. Replace it with the actual contract address.

### **Step 3: Call the Contract Function**
- The `call` method is used to invoke the `getTotalSupply` function on the contract.
- The result is returned as a `BigNumber` object, which is converted to a string using `toString()`.

---

## **4. Run the Script**

Save the script as `call_contract_function.php` and run it from the command line:

```bash
php call_contract_function.php
```

### **Expected Output**
If the contract address and ABI are correct, the script will output the total supply:

```
Total Supply: 1000000
```

---

## **5. Notes**
- **ABI**: Ensure the ABI matches the function you want to call. For example, if your function returns a `uint256`, the ABI should reflect that.
- **Infura**: If you’re using Infura, replace `YOUR_INFURA_PROJECT_ID` with your actual Infura project ID.
- **Hardhat**: If you’re using a local Hardhat node, ensure the node is running at `http://localhost:8545`.
- **Error Handling**: The script includes basic error handling. If the connection fails or the function call fails, an error message will be displayed.

---

## **6. Conclusion**

This PHP script demonstrates how to call a "view" function (e.g., `getTotalSupply`) from a custom smart contract using the `web3.php` library. You can extend this script to call other functions or integrate it into a Laravel application for more advanced use cases.