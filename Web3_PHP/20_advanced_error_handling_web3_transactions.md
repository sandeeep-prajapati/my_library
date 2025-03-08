### File Name: `20_advanced_error_handling_web3_transactions.md`

---

## **Step 1: Create a Web3 Transaction Service**
Run this command to create the service:

```bash
php artisan make:service Web3TransactionService
```

---

## **Step 2: Implement Transaction Logic with Retry Mechanism**
**`app/Services/Web3TransactionService.php`**
```php
<?php

namespace App\Services;

use Exception;
use Illuminate\Support\Facades\Log;
use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

class Web3TransactionService
{
    protected $web3;
    protected $maxRetries = 3; // Maximum retry attempts

    public function __construct()
    {
        $rpcUrl = env('WEB3_MAINNET_RPC');  // Default to Mainnet
        $this->web3 = new Web3(new HttpProvider(new HttpRequestManager($rpcUrl, 10)));
    }

    /**
     * Send a Web3 transaction with retry logic
     */
    public function sendTransaction(array $txData)
    {
        $retryCount = 0;

        while ($retryCount < $this->maxRetries) {
            try {
                $result = $this->processTransaction($txData);
                if ($result) {
                    return [
                        'success' => true,
                        'message' => 'Transaction successful',
                        'txHash'  => $result
                    ];
                }
            } catch (Exception $e) {
                Log::error("Transaction Error: {$e->getMessage()}");

                if ($this->isGasIssue($e)) {
                    Log::warning("Gas issue detected. Retrying... Attempt #{$retryCount}");
                    $txData['gasPrice'] = $this->increaseGasPrice($txData['gasPrice']);
                } else {
                    return [
                        'success' => false,
                        'error' => $e->getMessage(),
                    ];
                }
            }

            $retryCount++;
        }

        return [
            'success' => false,
            'error' => 'Max retry limit reached. Transaction failed.'
        ];
    }

    /**
     * Process the actual transaction
     */
    private function processTransaction(array $txData)
    {
        $txHash = null;
        $this->web3->eth->sendTransaction($txData, function ($err, $result) use (&$txHash) {
            if ($err !== null) {
                throw new Exception($err->getMessage());
            }
            $txHash = $result;
        });

        return $txHash;
    }

    /**
     * Identify if the error is related to gas
     */
    private function isGasIssue(Exception $e)
    {
        return str_contains($e->getMessage(), 'gas required exceeds allowance');
    }

    /**
     * Dynamically increase gas price for retries
     */
    private function increaseGasPrice($currentGasPrice)
    {
        $newGasPrice = (int) $currentGasPrice * 1.2; // Increase gas price by 20%
        Log::info("New Gas Price: $newGasPrice");
        return $newGasPrice;
    }
}
```

---

## **Step 3: Create a Controller to Test Transactions**
Run this command:

```bash
php artisan make:controller TransactionController
```

---

### **Step 4: Implement Controller Logic**
**`app/Http/Controllers/TransactionController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\Web3TransactionService;

class TransactionController extends Controller
{
    public function sendTransaction(Request $request)
    {
        $service = new Web3TransactionService();

        $txData = [
            'from' => $request->input('from'),
            'to' => $request->input('to'),
            'value' => '0x29a2241af62c0000', // 0.01 ETH in Wei
            'gas' => '21000',
            'gasPrice' => '20000000000', // 20 Gwei
        ];

        $result = $service->sendTransaction($txData);

        return response()->json($result);
    }
}
```

---

## **Step 5: Define Routes**
In `routes/web.php`:

```php
use App\Http\Controllers\TransactionController;

Route::post('/web3/transaction/send', [TransactionController::class, 'sendTransaction']);
```

---

## **Step 6: Configure `.env`**
Add this to your `.env` file:

```env
WEB3_MAINNET_RPC=https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID
```

---

## **Step 7: Testing the Integration**
### **Sample Request Using cURL**
```bash
curl -X POST -d "from=0xYourWalletAddress&to=0xRecipientAddress" http://localhost:8000/web3/transaction/send
```

### **Sample JSON Response**
✅ **Success Response:**
```json
{
    "success": true,
    "message": "Transaction successful",
    "txHash": "0x4d8b8f123456..."
}
```

❌ **Failure Response After 3 Retries:**
```json
{
    "success": false,
    "error": "Max retry limit reached. Transaction failed."
}
```

---

## **Step 8: Logging Transactions for Tracking**
Laravel’s default logging mechanism will track:

✅ Transaction Success  
✅ Gas-Related Errors  
✅ Retry Attempts  
✅ Final Failure After Max Retries  

---

## **Step 9: Bonus Enhancements**
✅ Implement notification alerts when retries fail (e.g., via Slack or email).  
✅ Add exponential backoff logic for more efficient retries.  
✅ Use Laravel Queue Jobs for handling bulk transactions in the background.  

---

If you'd like to extend this with real-time transaction status tracking or webhook updates, let me know! 🚀