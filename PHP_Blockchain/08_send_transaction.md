Here's a comprehensive guide to sending ERC-20 token transfers from Laravel using `web3.php`:

---

### **1. Prerequisites**
1. Install required packages:
   ```bash
   composer require web3/web3 php-http/guzzle7-adapter kornrunner/keccak web3p/ethereum-tx
   ```

2. Configure `.env`:
   ```env
   ETH_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
   CONTRACT_ADDRESS=0xYourERC20ContractAddress
   SENDER_PRIVATE_KEY=0xYourPrivateKeyNeverCommitThis
   CHAIN_ID=11155111  # Sepolia (1 for Mainnet)
   ```

---

### **2. Create Transaction Service**
```bash
php artisan make:service TokenService
```

**`app/Services/TokenService.php`**
```php
<?php

namespace App\Services;

use Web3\Web3;
use Web3\Contract;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;
use kornrunner\Keccak;
use Web3p\EthereumTx\Transaction;
use Exception;
use Illuminate\Support\Str;

class TokenService
{
    protected $web3;
    protected $contract;

    public function __construct()
    {
        $this->web3 = new Web3(new HttpProvider(
            new HttpRequestManager(config('blockchain.ethereum.rpc_url'), 30)
        ));

        $this->contract = new Contract($this->web3->provider, config('blockchain.contract_address'));
    }

    public function transferToken(string $to, float $amount, int $decimals = 18): string
    {
        try {
            // 1. Prepare transfer data
            $data = $this->createTransferData($to, $amount, $decimals);
            
            // 2. Build transaction
            $tx = [
                'nonce' => $this->getNonce(),
                'from' => $this->getSenderAddress(),
                'to' => config('blockchain.contract_address'),
                'gasPrice' => $this->getGasPrice(),
                'gasLimit' => '0x7A120', // 500,000 gas (adjust per token)
                'value' => '0x0',
                'data' => $data,
                'chainId' => config('blockchain.chain_id')
            ];

            // 3. Sign and send
            return $this->sendRawTransaction($tx);

        } catch (Exception $e) {
            throw new Exception("Token transfer failed: " . $e->getMessage());
        }
    }

    protected function createTransferData(string $to, float $amount, int $decimals): string
    {
        $this->contract->abi($this->getErc20Abi());
        
        $value = bcmul($amount, bcpow(10, $decimals));
        $params = [$to, $value];

        // Generate function signature
        $transferMethod = 'transfer(address,uint256)';
        $methodId = substr(Keccak::hash($transferMethod, 256), 0, 8);
        
        // Encode parameters
        $encodedParams = $this->contract->ethabi->encodeParameters(
            ['address', 'uint256'],
            $params
        );
        
        return '0x' . $methodId . $encodedParams;
    }

    protected function getNonce(): string
    {
        $nonce = 0;
        $this->web3->eth->getTransactionCount(
            $this->getSenderAddress(),
            'pending',
            function ($err, $result) use (&$nonce) {
                if ($err) throw new Exception($err->getMessage());
                $nonce = $result->toString();
            }
        );
        return '0x' . dechex($nonce);
    }

    protected function getGasPrice(): string
    {
        $gasPrice = '';
        $this->web3->eth->gasPrice(function ($err, $result) use (&$gasPrice) {
            if ($err) throw new Exception($err->getMessage());
            $gasPrice = '0x' . dechex(hexdec($result->toString()) * 1.2); // +20% buffer
        });
        return $gasPrice;
    }

    protected function sendRawTransaction(array $tx): string
    {
        $signedTx = (new Transaction($tx))->sign(env('SENDER_PRIVATE_KEY'));
        
        $txHash = '';
        $this->web3->eth->sendRawTransaction(
            '0x' . $signedTx,
            function ($err, $result) use (&$txHash) {
                if ($err) throw new Exception($err->getMessage());
                $txHash = $result;
            }
        );
        
        return $txHash;
    }

    protected function getSenderAddress(): string
    {
        return '0x' . substr(env('SENDER_PRIVATE_KEY'), -40);
    }

    protected function getErc20Abi(): string
    {
        return '[{
            "constant":false,
            "inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],
            "name":"transfer",
            "outputs":[{"name":"","type":"bool"}],
            "type":"function"
        }]';
    }
}
```

---

### **3. Create Controller**
```bash
php artisan make:controller TokenTransferController
```

**`app/Http/Controllers/TokenTransferController.php`**
```php
<?php

namespace App\Http\Controllers;

use App\Services\TokenService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class TokenTransferController extends Controller
{
    public function transfer(Request $request, TokenService $tokenService): JsonResponse
    {
        $request->validate([
            'to' => 'required|regex:/^0x[a-fA-F0-9]{40}$/',
            'amount' => 'required|numeric|min:0.000001',
            'decimals' => 'sometimes|integer|between:6,18'
        ]);

        try {
            $txHash = $tokenService->transferToken(
                $request->input('to'),
                $request->input('amount'),
                $request->input('decimals', 18)
            );

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

### **4. Add API Route**
**`routes/api.php`**
```php
use App\Http\Controllers\TokenTransferController;

Route::post('/tokens/transfer', [TokenTransferController::class, 'transfer']);
```

---

### **5. Execute Transfer**
```bash
curl -X POST http://your-app.test/api/tokens/transfer \
  -H "Content-Type: application/json" \
  -d '{"to": "0xRecipientAddress", "amount": 1.5}'
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
1. **Private Key Handling**
   - Never commit to version control
   - Use AWS Secrets Manager or Hashicorp Vault in production
   ```php
   // Alternative to .env
   $privateKey = config('vault.erc20_sender_key');
   ```

2. **Input Validation**
   ```php
   // In controller
   $request->validate([
       'to' => 'required|regex:/^0x[a-fA-F0-9]{40}$/',
       'amount' => 'required|numeric|min:0.000001'
   ]);
   ```

3. **Gas Estimation**
   ```php
   // More accurate gas estimation
   $this->web3->eth->estimateGas([
       'to' => $contractAddress,
       'data' => $data
   ], function ($err, $result) {
       $gasLimit = $result->toString();
   });
   ```

---

### **Advanced Features**
1. **Nonce Management**
   ```php
   // Track pending nonces in database
   DB::table('pending_txs')->insert([
       'nonce' => hexdec($tx['nonce']),
       'tx_hash' => $txHash
   ]);
   ```

2. **Transaction Status Monitoring**
   ```php
   $this->web3->eth->getTransactionReceipt($txHash, 
       function ($err, $receipt) {
           if ($receipt->status === '0x1') {
               // Success
           }
       }
   );
   ```

3. **Multi-chain Support**
   ```php
   // In config/blockchain.php
   'networks' => [
       'sepolia' => [
           'chain_id' => 11155111,
           'rpc_url' => env('SEPOLIA_RPC')
       ],
       'bsc' => [
           'chain_id' => 56,
           'rpc_url' => env('BSC_RPC')
       ]
   ]
   ```

---

### **Troubleshooting**
| Error | Solution |
|-------|----------|
| `insufficient funds` | Fund your sender address |
| `nonce too low` | Implement proper nonce tracking |
| `gas too low` | Increase gas limit or use estimation |
| `invalid sender` | Verify private key format (must start with 0x) |

---

### **Next Steps**
1. **Add webhook notifications** for transaction confirmations
2. **Implement batch transfers** for multiple recipients
3. **Add support for EIP-1559** transactions

Would you like me to elaborate on any of these advanced scenarios?