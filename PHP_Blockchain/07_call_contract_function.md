Here's a step-by-step guide to calling a **read-only Solidity function** (like `balanceOf`) from Laravel using `web3.php`:

---

### **1. Prerequisites**
1. Install required packages:
   ```bash
   composer require web3/web3 php-http/guzzle7-adapter
   ```

2. Store these in `.env`:
   ```env
   ETH_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
   CONTRACT_ADDRESS=0x123...  # Your contract address
   ```

---

### **2. Create Contract Helper Service**
```bash
php artisan make:service ContractService
```

**`app/Services/ContractService.php`**
```php
<?php

namespace App\Services;

use Web3\Contract;
use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

class ContractService
{
    protected $contract;
    protected $web3;

    public function __construct()
    {
        $this->web3 = new Web3(new HttpProvider(
            new HttpRequestManager(config('blockchain.ethereum.rpc_url'), 10)
        ));

        $this->contract = new Contract(
            $this->web3->provider,
            config('blockchain.contract_address')
        );
    }

    public function callReadFunction(string $abi, string $functionName, array $params = []): mixed
    {
        $this->contract->abi($abi);

        return new \React\Promise\Promise(function ($resolve, $reject) use ($functionName, $params) {
            $this->contract->call($functionName, ...$params, function ($err, $result) use ($resolve, $reject) {
                if ($err) {
                    $reject(new \Exception("Contract call failed: " . $err->getMessage()));
                    return;
                }
                $resolve($result);
            });
        });
    }
}
```

---

### **3. Create Controller for Balance Check**
```bash
php artisan make:controller BalanceController
```

**`app/Http/Controllers/BalanceController.php`**
```php
<?php

namespace App\Http\Controllers;

use App\Services\ContractService;
use Illuminate\Http\JsonResponse;

class BalanceController extends Controller
{
    public function getBalance(string $address, ContractService $contract): JsonResponse
    {
        try {
            // ERC-20 ABI snippet for balanceOf
            $abi = '[{
                "constant":true,
                "inputs":[{"name":"_owner","type":"address"}],
                "name":"balanceOf",
                "outputs":[{"name":"balance","type":"uint256"}],
                "type":"function"
            }]';

            $balance = $contract->callReadFunction($abi, 'balanceOf', [$address])
                ->then(function ($result) {
                    return $result[0]->toString(); // Returns BigNumber as string
                })
                ->wait(); // Wait for async call to complete

            return response()->json([
                'success' => true,
                'balance' => $balance,
                'normalized' => $this->weiToEth($balance),
                'address' => $address,
                'contract' => config('blockchain.contract_address')
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }

    protected function weiToEth(string $wei): float
    {
        return $wei / 1e18; // Adjust for token decimals if needed
    }
}
```

---

### **4. Add API Route**
**`routes/api.php`**
```php
use App\Http\Controllers\BalanceController;

Route::get('/balance/{address}', [BalanceController::class, 'getBalance']);
```

---

### **5. Call the Endpoint**
```bash
curl http://your-app.test/api/balance/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

**Successful Response:**
```json
{
    "success": true,
    "balance": "5000000000000000000", // 5 tokens in wei
    "normalized": 5.0,
    "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
    "contract": "0x123..."
}
```

---

### **Key Components Explained**
1. **ABI Definition**:
   - Only include the necessary function (reduces payload size)
   - For ERC-20, you can use the full ABI from [OpenZeppelin](https://docs.openzeppelin.com/contracts/4.x/api/token/erc20)

2. **Asynchronous Handling**:
   - Uses ReactPHP promises
   - `->wait()` blocks until response is received

3. **Unit Conversion**:
   - `weiToEth()` converts raw wei to human-readable format
   - For tokens with different decimals, adjust the divisor:
     ```php
     // For USDC (6 decimals):
     return $wei / 1e6;
     ```

---

### **Advanced: Caching Results**
Modify `ContractService`:
```php
use Illuminate\Support\Facades\Cache;

public function cachedCall(string $abi, string $function, array $params = [], int $ttl = 60)
{
    $cacheKey = md5("contract_call:$function:" . implode(',', $params));

    return Cache::remember($cacheKey, $ttl, function() use ($abi, $function, $params) {
        return $this->callReadFunction($abi, $function, $params)->wait();
    });
}
```

---

### **Error Handling Scenarios**
| Error | Solution |
|-------|----------|
| `Invalid ABI` | Verify ABI matches contract |
| `Invalid address` | Check address checksum |
| `Contract not deployed` | Verify contract address |
| `Node timeout` | Increase timeout in `HttpRequestManager` |

---

### **Testing with Tinker**
```bash
php artisan tinker
>>> $contract = app(App\Services\ContractService::class)
>>> $contract->callReadFunction($abi, 'balanceOf', ['0x742...'])->wait()
```

---

### **Next Steps**
1. **Add support for**:
   - Multiple contracts
   - Batch requests
   - Event listening

2. **Implement**:
   ```php
   // Get token symbol
   $contract->callReadFunction($abi, 'symbol')->wait();

   // Get token decimals
   $contract->callReadFunction($abi, 'decimals')->wait();
   ```
