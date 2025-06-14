Here's a complete implementation to fetch and display real-time gas prices (low, medium, high) in Laravel:

### 1. Install Required Packages
```bash
composer require web3/web3 php-http/guzzle7-adapter
```

### 2. Create Gas Service
```bash
php artisan make:service GasService
```

**`app/Services/GasService.php`**
```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Illuminate\Support\Facades\Cache;
use Exception;

class GasService
{
    protected $web3;
    protected $cacheKey = 'gas_prices';
    protected $cacheTtl = 15; // seconds

    public function __construct()
    {
        $this->web3 = new Web3(new HttpProvider(
            new HttpRequestManager(config('blockchain.ethereum.rpc_url'), 10)
        ));
    }

    public function getGasPrices(): array
    {
        return Cache::remember($this->cacheKey, $this->cacheTtl, function () {
            try {
                $gasPrices = [];
                
                // Fetch current gas price
                $this->web3->eth->gasPrice(function ($err, $result) use (&$gasPrices) {
                    if ($err) throw new Exception($err->getMessage());
                    $baseFee = hexdec($result->toString());
                    
                    // Calculate tiers (these multipliers are estimates)
                    $gasPrices = [
                        'low' => $baseFee * 0.9,
                        'medium' => $baseFee * 1.0,
                        'high' => $baseFee * 1.1,
                        'base_fee' => $baseFee,
                        'unit' => 'wei',
                        'timestamp' => now()->toDateTimeString()
                    ];
                });

                return $gasPrices;

            } catch (Exception $e) {
                throw new Exception("Failed to fetch gas prices: " . $e->getMessage());
            }
        });
    }

    public function formatForDisplay(array $gasPrices): array
    {
        return [
            'low' => $this->weiToGwei($gasPrices['low']),
            'medium' => $this->weiToGwei($gasPrices['medium']),
            'high' => $this->weiToGwei($gasPrices['high']),
            'base_fee' => $this->weiToGwei($gasPrices['base_fee']),
            'unit' => 'gwei',
            'timestamp' => $gasPrices['timestamp']
        ];
    }

    protected function weiToGwei($wei): float
    {
        return round($wei / 1000000000, 2); // 1 Gwei = 10^9 Wei
    }
}
```

### 3. Create Controller
```bash
php artisan make:controller GasController
```

**`app/Http/Controllers/GasController.php`**
```php
<?php

namespace App\Http\Controllers;

use App\Services\GasService;
use Illuminate\Http\JsonResponse;

class GasController extends Controller
{
    public function index(GasService $gasService): JsonResponse
    {
        try {
            $gasPrices = $gasService->getGasPrices();
            return response()->json([
                'success' => true,
                'data' => $gasService->formatForDisplay($gasPrices)
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
```

### 4. Add API Route
**`routes/api.php`**
```php
use App\Http\Controllers\GasController;

Route::get('/gas-prices', [GasController::class, 'index']);
```

### 5. Create Artisan Command
```bash
php artisan make:command GetGasPrices
```

**`app/Console/Commands/GetGasPrices.php`**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\GasService;

class GetGasPrices extends Command
{
    protected $signature = 'gas:prices';
    protected $description = 'Get current Ethereum gas prices';

    public function handle(GasService $gasService)
    {
        try {
            $prices = $gasService->getGasPrices();
            $display = $gasService->formatForDisplay($prices);

            $this->table(
                ['Tier', 'Price (Gwei)'],
                [
                    ['Low', $display['low']],
                    ['Medium', $display['medium']],
                    ['High', $display['high']],
                    ['Base Fee', $display['base_fee']]
                ]
            );

            $this->line("Last updated: {$display['timestamp']}");

        } catch (\Exception $e) {
            $this->error($e->getMessage());
            return 1;
        }

        return 0;
    }
}
```

### 6. Usage Examples

**API Endpoint:**
```bash
curl http://your-app.test/api/gas-prices
```
Sample response:
```json
{
    "success": true,
    "data": {
        "low": 12.34,
        "medium": 15.67,
        "high": 18.90,
        "base_fee": 15.67,
        "unit": "gwei",
        "timestamp": "2023-10-26 09:30:45"
    }
}
```

**Artisan Command:**
```bash
php artisan gas:prices
```
Sample output:
```
+---------+-------------+
| Tier    | Price (Gwei)|
+---------+-------------+
| Low     | 12.34       |
| Medium  | 15.67       |
| High    | 18.90       |
| Base Fee| 15.67       |
+---------+-------------+
Last updated: 2023-10-26 09:30:45
```

### Key Features:

1. **Caching**: Prices cached for 15 seconds to avoid rate limits
2. **Multiple Access Methods**: Both API and CLI interfaces
3. **Unit Conversion**: Automatically converts wei → gwei
4. **Error Handling**: Graceful failure for RPC issues
5. **Tiered Pricing**: Calculates low/medium/high estimates

### Advanced Enhancement (EIP-1559):

For modern gas price estimation with base fee and priority fee:

```php
public function getEip1559GasPrices(): array
{
    $this->web3->eth->getBlock('pending', true, function ($err, $block) {
        $baseFee = hexdec($block->baseFeePerGas->toString());
        // Calculate priority fees (historical percentiles)
        // Requires additional provider support
    });
}
```

Would you like me to add:
1. Historical gas price tracking?
2. EIP-1559 specific implementation?
3. Notification system when prices drop below threshold?