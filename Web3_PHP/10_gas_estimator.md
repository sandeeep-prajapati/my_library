Here’s a Markdown file (`10_gas_estimator.md`) that explains how to create a gas estimator function in Web3 PHP to calculate the gas required for sending 0.1 ETH.

---

# Gas Estimator Function in Web3 PHP

This guide demonstrates how to create a gas estimator function in Web3 PHP to calculate the gas required for sending 0.1 ETH.

---

## **1. Install Web3 PHP Library**

First, install the `web3.php` library using Composer. This library allows you to interact with Ethereum nodes via JSON-RPC.

```bash
composer require web3/web3
```

---

## **2. Create the Gas Estimator Function**

Create a function to estimate the gas required for sending 0.1 ETH.

### **Step 1: Create the Function**
Create a new file `app/Services/GasEstimatorService.php`:

```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Web3\Utils;

class GasEstimatorService
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
     * Estimate the gas required for sending 0.1 ETH.
     *
     * @param string $fromAddress
     * @param string $toAddress
     * @return array
     */
    public function estimateGasForEthTransfer(string $fromAddress, string $toAddress): array
    {
        // Convert 0.1 ETH to Wei
        $amountInWei = Utils::toWei('0.1', 'ether');

        // Create the transaction object
        $transaction = [
            'from' => $fromAddress,
            'to' => $toAddress,
            'value' => '0x' . $amountInWei->toHex(),
        ];

        // Estimate the gas
        $gasEstimate = 0;
        $this->web3->eth->estimateGas($transaction, function ($err, $result) use (&$gasEstimate) {
            if ($err !== null) {
                throw new \Exception("Error estimating gas: " . $err->getMessage());
            }
            $gasEstimate = $result->toString();
        });

        // Get the current gas price
        $gasPrice = '0';
        $this->web3->eth->gasPrice(function ($err, $result) use (&$gasPrice) {
            if ($err !== null) {
                throw new \Exception("Error fetching gas price: " . $err->getMessage());
            }
            $gasPrice = $result->toString();
        });

        // Calculate the total gas cost in Wei
        $totalGasCostInWei = gmp_mul($gasEstimate, $gasPrice);

        // Convert the total gas cost to ETH
        $totalGasCostInEth = Utils::fromWei($totalGasCostInWei, 'ether');

        return [
            'gas_estimate' => $gasEstimate,
            'gas_price' => $gasPrice,
            'total_gas_cost_wei' => $totalGasCostInWei,
            'total_gas_cost_eth' => $totalGasCostInEth,
        ];
    }
}
```

---

## **3. Use the Function in a Laravel Controller**

Create a controller to handle the request and return the gas estimate.

### **Step 1: Create the Controller**
Run the following Artisan command:

```bash
php artisan make:controller GasEstimatorController
```

### **Step 2: Implement the Controller Logic**
Update the `app/Http/Controllers/GasEstimatorController.php` file:

```php
<?php

namespace App\Http\Controllers;

use App\Services\GasEstimatorService;
use Illuminate\Http\Request;

class GasEstimatorController extends Controller
{
    protected $gasEstimatorService;

    public function __construct(GasEstimatorService $gasEstimatorService)
    {
        $this->gasEstimatorService = $gasEstimatorService;
    }

    /**
     * Estimate the gas required for sending 0.1 ETH.
     *
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function estimateGas(Request $request)
    {
        // Validate the request
        $request->validate([
            'from_address' => 'required|string|regex:/^0x[a-fA-F0-9]{40}$/',
            'to_address' => 'required|string|regex:/^0x[a-fA-F0-9]{40}$/',
        ]);

        // Fetch the input data
        $fromAddress = $request->query('from_address');
        $toAddress = $request->query('to_address');

        // Estimate the gas
        $gasEstimate = $this->gasEstimatorService->estimateGasForEthTransfer($fromAddress, $toAddress);

        // Return the gas estimate in the response
        return response()->json($gasEstimate);
    }
}
```

---

## **4. Define the API Route**

Add a route to `routes/api.php` to expose the API endpoint:

```php
use App\Http\Controllers\GasEstimatorController;

Route::get('/gas-estimate', [GasEstimatorController::class, 'estimateGas']);
```

---

## **5. Test the API**

Start your Laravel development server:

```bash
php artisan serve
```

### **Example Request**
Make a GET request to the API endpoint with the `from_address` and `to_address` as query parameters:

```
http://localhost:8000/api/gas-estimate?from_address=0xYourFromAddress&to_address=0xYourToAddress
```

### **Example Response**
If the addresses are valid, the API will return the gas estimate and total gas cost:

```json
{
    "gas_estimate": "21000",
    "gas_price": "20000000000",
    "total_gas_cost_wei": "420000000000000",
    "total_gas_cost_eth": "0.00042"
}
```

---

## **6. Notes**
- **Infura**: If you’re using Infura, replace `YOUR_INFURA_PROJECT_ID` with your actual Infura project ID.
- **Hardhat**: If you’re using a local Hardhat node, ensure the node is running at `http://localhost:8545`.
- **Error Handling**: The API includes basic validation and error handling. If the addresses are invalid or the gas estimation fails, an appropriate error message will be returned.

---

## **7. Conclusion**

This Laravel API allows you to estimate the gas required for sending 0.1 ETH using the `web3.php` library. You can extend this functionality to include additional features, such as estimating gas for other types of transactions or integrating with a wallet service.