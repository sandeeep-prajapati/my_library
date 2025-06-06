Here’s a Markdown file (`05_eth_balance_api.md`) that explains how to create a Laravel API to retrieve the ETH balance of a given Ethereum address using the `web3.php` library.

---

# Laravel API to Retrieve ETH Balance of an Ethereum Address

This guide demonstrates how to create a Laravel API that accepts an Ethereum address and returns its ETH balance using the `web3.php` library.

---

## **1. Install Web3 PHP Library**

First, install the `web3.php` library using Composer. This library allows you to interact with Ethereum nodes via JSON-RPC.

```bash
composer require web3/web3
```

---

## **2. Create a Service Class for Ethereum Interactions**

Create a service class to handle Ethereum interactions. This class will encapsulate the logic for fetching the ETH balance.

### **Step 1: Create the Service Class**
Create a new file `app/Services/EthereumService.php`:

```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

class EthereumService
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
     * Get the ETH balance of a given Ethereum address.
     *
     * @param string $address
     * @return string
     */
    public function getBalance(string $address): string
    {
        $balance = '0';
        $this->web3->eth->getBalance($address, function ($err, $result) use (&$balance) {
            if ($err !== null) {
                throw new \Exception("Error fetching balance: " . $err->getMessage());
            }
            $balance = $result->toString();
        });

        return $balance;
    }
}
```

---

## **3. Create a Laravel Controller**

Create a controller to handle the API request and return the ETH balance.

### **Step 1: Create the Controller**
Run the following Artisan command to create a controller:

```bash
php artisan make:controller EthereumController
```

### **Step 2: Implement the Controller Logic**
Update the `app/Http/Controllers/EthereumController.php` file:

```php
<?php

namespace App\Http\Controllers;

use App\Services\EthereumService;
use Illuminate\Http\Request;

class EthereumController extends Controller
{
    protected $ethereumService;

    public function __construct(EthereumService $ethereumService)
    {
        $this->ethereumService = $ethereumService;
    }

    /**
     * Get the ETH balance of a given Ethereum address.
     *
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function getBalance(Request $request)
    {
        // Validate the request
        $request->validate([
            'address' => 'required|string|regex:/^0x[a-fA-F0-9]{40}$/',
        ]);

        // Fetch the balance
        $address = $request->query('address');
        $balance = $this->ethereumService->getBalance($address);

        // Return the balance in the response
        return response()->json([
            'address' => $address,
            'balance' => $balance,
        ]);
    }
}
```

---

## **4. Define the API Route**

Add a route to `routes/api.php` to expose the API endpoint:

```php
use App\Http\Controllers\EthereumController;

Route::get('/balance', [EthereumController::class, 'getBalance']);
```

---

## **5. Test the API**

Start your Laravel development server:

```bash
php artisan serve
```

### **Example Request**
Make a GET request to the API endpoint with an Ethereum address as a query parameter:

```
http://localhost:8000/api/balance?address=0xYourEthereumAddress
```

### **Example Response**
If the address is valid, the API will return a JSON response with the ETH balance:

```json
{
    "address": "0xYourEthereumAddress",
    "balance": "1000000000000000000" // Balance in Wei
}
```

---

## **6. Notes**
- **Infura**: If you’re using Infura, replace `YOUR_INFURA_PROJECT_ID` with your actual Infura project ID.
- **Hardhat**: If you’re using a local Hardhat node, ensure the node is running at `http://localhost:8545`.
- **Wei to ETH Conversion**: The balance is returned in Wei (the smallest unit of ETH). To convert it to ETH, divide by `10^18`.
- **Error Handling**: The API includes basic validation and error handling. If the address is invalid or the request fails, an appropriate error message will be returned.

---

## **7. Conclusion**

This Laravel API allows you to retrieve the ETH balance of a given Ethereum address using the `web3.php` library. You can extend this API to include additional features, such as converting the balance to ETH or supporting multiple networks.