### File Name: `18_switch_networks_in_laravel.md`

---

## **Step 1: Update `.env` for Network Configuration**
Add the following entries to your `.env` file for dynamic network switching:

```env
# Ethereum Mainnet
WEB3_NETWORK=mainnet
WEB3_MAINNET_RPC=https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID

# Goerli Testnet
WEB3_GOERLI_RPC=https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID

# Polygon Mainnet
WEB3_POLYGON_RPC=https://polygon-rpc.com

# Binance Smart Chain (BSC) Mainnet
WEB3_BSC_RPC=https://bsc-dataseed.binance.org/
```

---

## **Step 2: Create a Network Configuration Service**
Run this command to generate a new service class:

```bash
php artisan make:service NetworkService
```

---

### **Step 3: Implement Network Switching Logic**
**`app/Services/NetworkService.php`**
```php
<?php

namespace App\Services;

use Web3\Web3;

class NetworkService
{
    protected $web3;

    public function __construct()
    {
        $this->initializeWeb3();
    }

    // Initialize Web3 with appropriate network
    private function initializeWeb3()
    {
        $network = env('WEB3_NETWORK', 'mainnet');  // Default to Ethereum Mainnet

        $rpcUrl = match ($network) {
            'mainnet' => env('WEB3_MAINNET_RPC'),
            'goerli' => env('WEB3_GOERLI_RPC'),
            'polygon' => env('WEB3_POLYGON_RPC'),
            'bsc' => env('WEB3_BSC_RPC'),
            default => env('WEB3_MAINNET_RPC'), // Fallback to Mainnet
        };

        $this->web3 = new Web3($rpcUrl);
    }

    // Example: Get Latest Block Number
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

    // Example: Get Current Network Info
    public function getNetworkInfo()
    {
        return [
            'network' => env('WEB3_NETWORK'),
            'rpc_url' => $this->web3->getProvider()->getHost(),
        ];
    }
}
```

---

## **Step 4: Create a Controller for Network Handling**
Run this command:

```bash
php artisan make:controller NetworkController
```

---

### **Step 5: Implement Controller Logic**
**`app/Http/Controllers/NetworkController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\NetworkService;

class NetworkController extends Controller
{
    protected $networkService;

    public function __construct(NetworkService $networkService)
    {
        $this->networkService = $networkService;
    }

    public function getNetworkInfo()
    {
        return response()->json($this->networkService->getNetworkInfo());
    }

    public function getLatestBlock()
    {
        return response()->json([
            'latest_block' => $this->networkService->getLatestBlockNumber()
        ]);
    }
}
```

---

## **Step 6: Add Routes**
In `routes/web.php`, add the following routes:

```php
use App\Http\Controllers\NetworkController;

Route::get('/network/info', [NetworkController::class, 'getNetworkInfo']);
Route::get('/network/block/latest', [NetworkController::class, 'getLatestBlock']);
```

---

## **Step 7: Testing the Integration**
### **1. Switch Network (via `.env` file)**
Change the `WEB3_NETWORK` value in `.env`:

```env
WEB3_NETWORK=polygon
```

Run the following command to refresh the environment variables:

```bash
php artisan config:cache
```

### **2. Test Endpoints**
- **Get Current Network Info**
```bash
curl http://localhost:8000/network/info
```
**Response:**
```json
{
    "network": "polygon",
    "rpc_url": "https://polygon-rpc.com"
}
```

- **Get Latest Block**
```bash
curl http://localhost:8000/network/block/latest
```
**Response Example:**
```json
{
    "latest_block": "4865273"
}
```

---

## **Step 8: Bonus Enhancements for Robustness**
✅ Add a middleware to restrict network switching to admin users only.  
✅ Use a dynamic dropdown in your frontend for seamless network selection.  
✅ Implement error handling for RPC connectivity issues.  

---

If you'd like guidance on adding MetaMask integration or Web3 wallet connection, let me know! 🚀