
---

### **1. Install Required Packages**
```bash
composer require web3/web3 php-http/guzzle7-adapter
```

---

### **2. Configure Environment (`.env`)**
```env
ETH_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
ETH_RPC_TIMEOUT=10  # Seconds
ETH_RPC_RATE_LIMIT=30  # Max requests per minute
```

---

### **3. Create Web3 Service**
```bash
php artisan make:service EthereumService
```

**`app/Services/EthereumService.php`**
```php
use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;

class EthereumService
{
    protected $web3;
    protected $rateLimitKey = 'eth_rpc_rate_limit';

    public function __construct()
    {
        $provider = new HttpProvider(
            new HttpRequestManager(
                config('blockchain.ethereum.rpc_url'),
                config('blockchain.ethereum.timeout')
            )
        );
        $this->web3 = new Web3($provider);
    }

    public function getLatestBlockNumber(): array
    {
        // Rate limiting
        if (Cache::has($this->rateLimitKey)) {
            $attempts = Cache::get($this->rateLimitKey);
            if ($attempts >= config('blockchain.ethereum.rate_limit')) {
                throw new \Exception('Rate limit exceeded', 429);
            }
        }

        try {
            $blockNumber = null;
            $this->web3->eth->blockNumber(function ($err, $block) use (&$blockNumber) {
                if ($err) throw new \Exception($err->getMessage());
                $blockNumber = hexdec($block->toString());
            });

            // Increment rate limit counter
            Cache::add($this->rateLimitKey, 1, now()->addMinutes(1));
            Cache::increment($this->rateLimitKey);

            return [
                'success' => true,
                'block_number' => $blockNumber,
                'timestamp' => now()->toDateTimeString()
            ];

        } catch (\Exception $e) {
            Log::error("Web3 Error: " . $e->getMessage());
            return [
                'success' => false,
                'error' => $this->formatError($e),
                'retry_after' => $this->getRetryTime($e)
            ];
        }
    }

    protected function formatError(\Exception $e): array
    {
        return [
            'message' => $e->getMessage(),
            'code' => $e->getCode(),
            'type' => get_class($e)
        ];
    }

    protected function getRetryTime(\Exception $e): ?int
    {
        return match ($e->getCode()) {
            429 => 60, // Seconds
            default => null
        };
    }
}
```

---

### **4. Create Controller**
```bash
php artisan make:controller EthereumController
```

**`app/Http/Controllers/EthereumController.php`**
```php
use App\Services\EthereumService;
use Illuminate\Http\JsonResponse;

class EthereumController extends Controller
{
    public function getBlockNumber(EthereumService $ethereum): JsonResponse
    {
        $response = $ethereum->getLatestBlockNumber();
        
        return response()->json(
            data: $response,
            status: $response['success'] ? 200 : ($response['error']['code'] ?? 500)
        );
    }
}
```

---

### **5. Add API Route**
**`routes/api.php`**
```php
use App\Http\Controllers\EthereumController;

Route::get('/eth/block-number', [EthereumController::class, 'getBlockNumber'])
     ->middleware('throttle:30,1'); // Additional Laravel-level rate limiting
```

---

### **6. Test the Endpoint**
```bash
curl http://your-app.test/api/eth/block-number
```

**Successful Response:**
```json
{
    "success": true,
    "block_number": 19238765,
    "timestamp": "2023-10-25 14:30:22"
}
```

**Error Responses:**
```json
{
    "success": false,
    "error": {
        "message": "Rate limit exceeded",
        "code": 429,
        "type": "Exception"
    },
    "retry_after": 60
}
```

```json
{
    "success": false,
    "error": {
        "message": "cURL error 28: Connection timed out",
        "code": 504,
        "type": "Exception"
    }
}
```

---

### **Key Features Implemented**
1. **Dual Rate Limiting**:
   - Application-level (Laravel throttle middleware)
   - RPC-level (Custom counter in cache)

2. **Error Handling**:
   - Timeouts (cURL/Guzzle)
   - Rate limits (429)
   - Invalid responses

3. **Logging**:
   - All errors logged to Laravel's default channel

4. **Retry Guidance**:
   - Returns `retry_after` seconds for rate limits

5. **Type Safety**:
   - Consistent JSON response structure

---

### **Optimization Tips**
1. **Cache Block Numbers**:
   ```php
   return Cache::remember('latest_block', 15, function() use ($ethereum) {
       return $ethereum->getLatestBlockNumber();
   });
   ```

2. **Circuit Breaker Pattern**:
   ```php
   if (Cache::get('rpc_circuit_open')) {
       throw new \Exception('Service temporarily unavailable', 503);
   }
   ```

3. **Queue Workers**:
   ```bash
   php artisan queue:work --queue=web3
   ```
   For non-critical requests

---

### **Next Steps**
1. **Add Monitoring**:
   ```php
   Log::info("Block fetched", ['block' => $blockNumber]);
   ```

2. **Extend to BSC**:
   ```php
   $this->web3 = new Web3(new HttpProvider(
       new HttpRequestManager(config('blockchain.bsc.rpc_url'))
   ));
   ```
