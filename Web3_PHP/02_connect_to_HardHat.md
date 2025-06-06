
---

# Ethereum Service in Laravel using Hardhat

This guide explains how to create a Laravel service class to connect to a local Ethereum node using Hardhat. It also covers the differences between **Infura** and **Hardhat**.

---

## **1. Laravel Service Class to Connect to a Local Ethereum Node**

### **Step 1: Install Required Packages**
To interact with Ethereum in Laravel, you need the `web3.php` library. Install it via Composer:

```bash
composer require web3/web3
```

### **Step 2: Create the Service Class**
Create a new service class in Laravel to handle Ethereum interactions.

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
        // Connect to the local Hardhat node (default RPC URL: http://localhost:8545)
        $provider = new HttpProvider(new HttpRequestManager('http://localhost:8545'));
        $this->web3 = new Web3($provider);
    }

    /**
     * Get the latest block number from the Ethereum node.
     *
     * @return int
     */
    public function getLatestBlockNumber(): int
    {
        $blockNumber = 0;
        $this->web3->eth->blockNumber(function ($err, $result) use (&$blockNumber) {
            if ($err !== null) {
                throw new \Exception("Error fetching block number: " . $err->getMessage());
            }
            $blockNumber = $result->toString();
        });

        return $blockNumber;
    }

    /**
     * Get the balance of an Ethereum address.
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

### **Step 3: Use the Service in a Controller**
You can now use the `EthereumService` in a Laravel controller.

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

    public function getBlockNumber()
    {
        $blockNumber = $this->ethereumService->getLatestBlockNumber();
        return response()->json(['block_number' => $blockNumber]);
    }

    public function getBalance(Request $request)
    {
        $address = $request->query('address');
        $balance = $this->ethereumService->getBalance($address);
        return response()->json(['balance' => $balance]);
    }
}
```

### **Step 4: Add Routes**
Add routes to your `routes/web.php` file to expose the endpoints.

```php
use App\Http\Controllers\EthereumController;

Route::get('/block-number', [EthereumController::class, 'getBlockNumber']);
Route::get('/balance', [EthereumController::class, 'getBalance']);
```

---

## **2. Differences Between Infura and Hardhat**

### **Infura**
- **Purpose**: Infura is a hosted Ethereum node service that provides access to Ethereum mainnet and testnets (e.g., Ropsten, Rinkeby, Goerli).
- **Use Case**:
  - Connect to Ethereum mainnet or testnets without running your own node.
  - Ideal for production applications or when you don’t want to manage node infrastructure.
- **Features**:
  - Scalable and reliable infrastructure.
  - Supports JSON-RPC and WebSocket APIs.
  - Free tier available with rate limits.
- **Limitations**:
  - Requires an API key.
  - Limited control over the node (e.g., no custom mining or forking).

### **Hardhat**
- **Purpose**: Hardhat is a development framework for Ethereum smart contracts. It includes a local Ethereum network for testing and development.
- **Use Case**:
  - Local development and testing of smart contracts.
  - Debugging with advanced features like stack traces and `console.log`.
  - Forking mainnet or testnets for realistic testing.
- **Features**:
  - Built-in local Ethereum network (Hardhat Network).
  - Advanced debugging tools.
  - Scriptable and extensible via plugins.
- **Limitations**:
  - Not suitable for production (local-only).
  - Requires manual setup for complex workflows.

### **Comparison Table**

| Feature                | Infura                          | Hardhat                          |
|------------------------|---------------------------------|----------------------------------|
| **Purpose**            | Hosted Ethereum node service   | Local development framework      |
| **Network**            | Mainnet and testnets           | Local network (Hardhat Network) |
| **Use Case**           | Production and testing         | Development and testing         |
| **Control**            | Limited (hosted service)       | Full control (local node)       |
| **Debugging Tools**    | No                             | Advanced (stack traces, etc.)   |
| **Forking**            | No                             | Yes                              |
| **Cost**               | Free tier, paid plans          | Free                            |

---

## **Conclusion**
- Use **Hardhat** for local development and testing of Ethereum smart contracts.
- Use **Infura** for connecting to Ethereum mainnet or testnets in production or when you don’t want to manage node infrastructure.

By combining both tools, you can create a robust development workflow: use Hardhat for local testing and Infura for deploying to live networks.