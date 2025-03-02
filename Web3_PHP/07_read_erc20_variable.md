Here’s a Markdown file (`07_read_erc20_variable.md`) that explains how to write a function to read a public variable (like `name` or `symbol`) from an ERC20 contract using Web3 PHP.

---

# Read a Public Variable from an ERC20 Contract using Web3 PHP

This guide demonstrates how to write a function to read a public variable (e.g., `name` or `symbol`) from an ERC20 contract using the `web3.php` library.

---

## **1. Install Web3 PHP Library**

First, install the `web3.php` library using Composer. This library allows you to interact with Ethereum nodes via JSON-RPC.

```bash
composer require web3/web3
```

---

## **2. Create a Function to Read ERC20 Contract Variables**

Create a function to interact with an ERC20 contract and read its public variables.

### **Step 1: Create the Function**
Create a new file `app/Services/Erc20Service.php`:

```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Web3\Contract;

class Erc20Service
{
    protected $web3;

    public function __construct()
    {
        // Replace with your Ethereum node URL (e.g., Infura or local Hardhat node)
        $nodeUrl = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'; // For Infura
        // $nodeUrl = 'http://localhost:8545'; // For local Hardhat node

        // Initialize Web3
        $this->web3 = new Web3(new HttpProvider(new HttpRequestManager($nodeUrl)));
    }

    /**
     * Read a public variable from an ERC20 contract.
     *
     * @param string $contractAddress
     * @param string $variableName
     * @return string
     */
    public function readVariable(string $contractAddress, string $variableName): string
    {
        // ERC20 ABI (simplified for name and symbol)
        $abi = '[{"constant":true,"inputs":[],"name":"' . $variableName . '","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"}]';

        // Initialize the contract
        $contract = new Contract($this->web3->provider, $abi);
        $contract->at($contractAddress);

        // Variable to store the result
        $result = '';

        // Call the contract function
        $contract->call($variableName, function ($err, $response) use (&$result) {
            if ($err !== null) {
                throw new \Exception("Error reading variable: " . $err->getMessage());
            }
            $result = $response[0];
        });

        return $result;
    }
}
```

---

## **3. Use the Function in a Laravel Controller**

Create a controller to handle the request and return the variable value.

### **Step 1: Create the Controller**
Run the following Artisan command:

```bash
php artisan make:controller Erc20Controller
```

### **Step 2: Implement the Controller Logic**
Update the `app/Http/Controllers/Erc20Controller.php` file:

```php
<?php

namespace App\Http\Controllers;

use App\Services\Erc20Service;
use Illuminate\Http\Request;

class Erc20Controller extends Controller
{
    protected $erc20Service;

    public function __construct(Erc20Service $erc20Service)
    {
        $this->erc20Service = $erc20Service;
    }

    /**
     * Read a public variable from an ERC20 contract.
     *
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function readVariable(Request $request)
    {
        // Validate the request
        $request->validate([
            'contract_address' => 'required|string|regex:/^0x[a-fA-F0-9]{40}$/',
            'variable_name' => 'required|string|in:name,symbol',
        ]);

        // Fetch the variable value
        $contractAddress = $request->query('contract_address');
        $variableName = $request->query('variable_name');
        $value = $this->erc20Service->readVariable($contractAddress, $variableName);

        // Return the value in the response
        return response()->json([
            'contract_address' => $contractAddress,
            'variable_name' => $variableName,
            'value' => $value,
        ]);
    }
}
```

---

## **4. Define the API Route**

Add a route to `routes/api.php` to expose the API endpoint:

```php
use App\Http\Controllers\Erc20Controller;

Route::get('/erc20/variable', [Erc20Controller::class, 'readVariable']);
```

---

## **5. Test the API**

Start your Laravel development server:

```bash
php artisan serve
```

### **Example Request**
Make a GET request to the API endpoint with the contract address and variable name as query parameters:

```
http://localhost:8000/api/erc20/variable?contract_address=0xYourContractAddress&variable_name=name
```

### **Example Response**
If the contract address and variable name are valid, the API will return the variable value:

```json
{
    "contract_address": "0xYourContractAddress",
    "variable_name": "name",
    "value": "YourTokenName"
}
```

---

## **6. Notes**
- **ABI**: The ABI provided in the example is simplified for reading `name` and `symbol`. For full functionality, use the complete ERC20 ABI.
- **Infura**: If you’re using Infura, replace `YOUR_INFURA_PROJECT_ID` with your actual Infura project ID.
- **Hardhat**: If you’re using a local Hardhat node, ensure the node is running at `http://localhost:8545`.
- **Error Handling**: The API includes basic validation and error handling. If the contract address or variable name is invalid, an appropriate error message will be returned.

---

## **7. Conclusion**

This Laravel API allows you to read public variables (e.g., `name` or `symbol`) from an ERC20 contract using the `web3.php` library. You can extend this functionality to include additional features, such as reading other variables or interacting with other types of smart contracts.