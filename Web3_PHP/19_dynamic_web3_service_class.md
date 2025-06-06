### File Name: `19_dynamic_web3_service_class.md`

---

## **Step 1: Update `.env` with Network Configuration**
Add multiple network configurations in your `.env` file for flexibility:

```env
# Default Network
WEB3_NETWORK=mainnet

# Ethereum Mainnet
WEB3_MAINNET_RPC=https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID
WEB3_MAINNET_CHAIN_ID=1

# Goerli Testnet
WEB3_GOERLI_RPC=https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID
WEB3_GOERLI_CHAIN_ID=5

# Polygon Mainnet
WEB3_POLYGON_RPC=https://polygon-rpc.com
WEB3_POLYGON_CHAIN_ID=137

# Binance Smart Chain (BSC)
WEB3_BSC_RPC=https://bsc-dataseed.binance.org/
WEB3_BSC_CHAIN_ID=56
```

---

## **Step 2: Create a Dynamic Web3 Service**
Run this command:

```bash
php artisan make:service DynamicWeb3Service
```

---

## **Step 3: Implement the Service Logic**
**`app/Services/DynamicWeb3Service.php`**
```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

class DynamicWeb3Service
{
    protected $web3;

    public function __construct(string $network = null)
    {
        $this->initializeWeb3($network ?? env('WEB3_NETWORK'));
    }

    // Initialize Web3 with dynamic network configuration
    private function initializeWeb3($network)
    {
        $rpcUrl = $this->getRpcUrl($network);
        $chainId = $this->getChainId($network);

        if (!$rpcUrl) {
            throw new \Exception("Invalid network: $network");
        }

        $this->web3 = new Web3(new HttpProvider(new HttpRequestManager($rpcUrl, 10)));
    }

    // Fetch RPC URL dynamically
    private function getRpcUrl($network)
    {
        return match ($network) {
            'mainnet' => env('WEB3_MAINNET_RPC'),
            'goerli' => env('WEB3_GOERLI_RPC'),
            'polygon' => env('WEB3_POLYGON_RPC'),
            'bsc' => env('WEB3_BSC_RPC'),
            default => null,
        };
    }

    // Fetch Chain ID dynamically
    private function getChainId($network)
    {
        return match ($network) {
            'mainnet' => env('WEB3_MAINNET_CHAIN_ID'),
            'goerli' => env('WEB3_GOERLI_CHAIN_ID'),
            'polygon' => env('WEB3_POLYGON_CHAIN_ID'),
            'bsc' => env('WEB3_BSC_CHAIN_ID'),
            default => null,
        };
    }

    // Example: Get Current Block Number
    public function getLatestBlockNumber()
    {
        $blockNumber = 0;
        $this->web3->eth->getBlockNumber(function ($err, $result) use (&$blockNumber) {
            if ($err === null) {
                $blockNumber = $result->toString();
            }
        });

        return $blockNumber;
    }

    // Example: Get Chain ID
    public function getChainInfo()
    {
        return [
            'rpc_url' => $this->web3->getProvider()->getHost(),
            'chain_id' => $this->getChainId(env('WEB3_NETWORK')),
        ];
    }
}
```

---

## **Step 4: Create a Controller for Testing**
Run this command:

```bash
php artisan make:controller Web3Controller
```

---

### **Step 5: Implement Controller Logic**
**`app/Http/Controllers/Web3Controller.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\DynamicWeb3Service;

class Web3Controller extends Controller
{
    public function getChainInfo(Request $request)
    {
        $network = $request->query('network', env('WEB3_NETWORK'));
        $web3Service = new DynamicWeb3Service($network);

        return response()->json($web3Service->getChainInfo());
    }

    public function getLatestBlock(Request $request)
    {
        $network = $request->query('network', env('WEB3_NETWORK'));
        $web3Service = new DynamicWeb3Service($network);

        return response()->json([
            'network' => $network,
            'latest_block' => $web3Service->getLatestBlockNumber(),
        ]);
    }
}
```

---

## **Step 6: Define Routes**
In `routes/web.php`, add these endpoints:

```php
use App\Http\Controllers\Web3Controller;

Route::get('/web3/chain-info', [Web3Controller::class, 'getChainInfo']);
Route::get('/web3/block/latest', [Web3Controller::class, 'getLatestBlock']);
```

---

## **Step 7: Testing the Integration**
### **1. Test with Default Network**
```bash
curl http://localhost:8000/web3/chain-info
```
**Response:**
```json
{
    "rpc_url": "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID",
    "chain_id": "1"
}
```

### **2. Test Switching Network (via Query Parameter)**
```bash
curl http://localhost:8000/web3/block/latest?network=polygon
```
**Response:**
```json
{
    "network": "polygon",
    "latest_block": "5043210"
}
```

---

## **Step 8: Bonus Enhancements for Scalability**
✅ Add caching for improved performance (e.g., cache block numbers to reduce RPC requests).  
✅ Implement rate-limiting for RPC requests to prevent abuse.  
✅ Add a custom `NetworkMiddleware` to restrict invalid network values.  

---

If you’d like to add features like contract interaction, wallet integration, or transaction signing, let me know! 🚀