Here's a step-by-step guide to deploy a Solidity contract from Laravel using `web3.php`:

---

### **1. Prerequisites**
1. Install required packages:
   ```bash
   composer require web3/web3 php-http/guzzle7-adapter
   npm install -g solc  # Solidity compiler
   ```

2. Store these in `.env`:
   ```env
   ETH_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
   DEPLOYER_PRIVATE_KEY=0xYourPrivateKeyNeverCommitThisToGit
   ```

---

### **2. Create Contract & Get Bytecode/ABI**
1. Save this as `contracts/SimpleStorage.sol`:
   ```solidity
   // SPDX-License-Identifier: MIT
   pragma solidity ^0.8.0;

   contract SimpleStorage {
       uint256 public value;

       function set(uint256 _value) public {
           value = _value;
       }

       function get() public view returns (uint256) {
           return value;
       }
   }
   ```

2. Compile it:
   ```bash
   solc --bin --abi --optimize -o build contracts/SimpleStorage.sol
   ```
   This generates:
   - `build/SimpleStorage.bin` (bytecode)
   - `build/SimpleStorage.abi` (ABI)

---

### **3. Create Deployment Service**
```bash
php artisan make:service ContractDeployer
```

**`app/Services/ContractDeployer.php`**
```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use Web3\Contract;
use Illuminate\Support\Str;
use kornrunner\Keccak;

class ContractDeployer
{
    protected $web3;
    protected $privateKey;

    public function __construct()
    {
        $this->web3 = new Web3(new HttpProvider(
            new HttpRequestManager(config('blockchain.ethereum.rpc_url'), 30)
        ));
        $this->privateKey = env('DEPLOYER_PRIVATE_KEY');
    }

    public function deploy(string $bytecode, string $abi): string
    {
        // 1. Get nonce
        $nonce = $this->getNonce();
        
        // 2. Build transaction
        $tx = [
            'nonce' => '0x' . dechex($nonce),
            'gasPrice' => '0x' . dechex(hexdec($this->getGasPrice()) * 1.1), // +10% buffer
            'gasLimit' => '0x47E7C4', // 4,700,000 gas (adjust per contract)
            'data' => '0x' . $bytecode,
        ];

        // 3. Sign transaction
        $signedTx = $this->signTransaction($tx);

        // 4. Send raw transaction
        $txHash = '';
        $this->web3->eth->sendRawTransaction('0x' . $signedTx, function ($err, $result) use (&$txHash) {
            if ($err) throw new \Exception("Deployment failed: " . $err->getMessage());
            $txHash = $result;
        });

        return $txHash;
    }

    protected function getNonce(): int
    {
        $address = '0x' . substr($this->privateKey, -40);
        $nonce = 0;
        
        $this->web3->eth->getTransactionCount($address, 'pending', function ($err, $result) use (&$nonce) {
            if ($err) throw new \Exception("Nonce fetch failed: " . $err->getMessage());
            $nonce = hexdec($result->toString());
        });

        return $nonce;
    }

    protected function getGasPrice(): string
    {
        $gasPrice = '';
        $this->web3->eth->gasPrice(function ($err, $result) use (&$gasPrice) {
            if ($err) throw new \Exception("Gas price fetch failed: " . $err->getMessage());
            $gasPrice = $result->toString();
        });
        return $gasPrice;
    }

    protected function signTransaction(array $tx): string
    {
        $transaction = new \Web3p\EthereumTx\Transaction($tx);
        return $transaction->sign($this->privateKey);
    }
}
```

---

### **4. Create Controller**
```bash
php artisan make:controller ContractController
```

**`app/Http/Controllers/ContractController.php`**
```php
<?php

namespace App\Http\Controllers;

use App\Services\ContractDeployer;
use Illuminate\Http\JsonResponse;

class ContractController extends Controller
{
    public function deploySimpleStorage(ContractDeployer $deployer): JsonResponse
    {
        try {
            $bytecode = file_get_contents(base_path('build/SimpleStorage.bin'));
            $abi = file_get_contents(base_path('build/SimpleStorage.abi'));
            
            $txHash = $deployer->deploy($bytecode, $abi);
            
            return response()->json([
                'success' => true,
                'tx_hash' => $txHash,
                'explorer_link' => 'https://sepolia.etherscan.io/tx/' . $txHash
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

---

### **5. Add Route**
**`routes/api.php`**
```php
use App\Http\Controllers\ContractController;

Route::post('/contracts/deploy', [ContractController::class, 'deploySimpleStorage']);
```

---

### **6. Deploy via API**
```bash
curl -X POST http://your-app.test/api/contracts/deploy
```

**Successful Response:**
```json
{
    "success": true,
    "tx_hash": "0x3fc...",
    "explorer_link": "https://sepolia.etherscan.io/tx/0x3fc..."
}
```

---

### **Key Security Practices**
1. **Never hardcode private keys** - Always use `.env`
2. **Use environment-specific keys** - Different keys for testnet/mainnet
3. **Validate bytecode** - Checksum verification before deployment
4. **Gas limit estimation** - Calculate properly to avoid failed deployments

---

### **Advanced: Verify on Etherscan**
Add this to `ContractDeployer`:
```php
public function verify(string $contractAddress, string $contractName)
{
    $exec = sprintf(
        'npx hardhat verify --network sepolia %s "%s"',
        $contractAddress,
        $contractName
    );
    
    return shell_exec($exec);
}
```
Call after deployment.

---

### **Troubleshooting**
| Error | Solution |
|-------|----------|
| `Invalid sender` | Check private key format (must start with 0x) |
| `Insufficient funds` | Fund your deployer address with test ETH |
| `Gas too low` | Increase gas limit in deployment transaction |
| `Invalid bytecode` | Recompile contract with `solc` |

---