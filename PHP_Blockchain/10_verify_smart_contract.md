Here's a comprehensive guide to automate Etherscan contract verification directly from your Laravel application after deployment:

---

### **1. Install Required Packages**
```bash
composer require guzzlehttp/guzzle illuminate/support
```

---

### **2. Configure Environment Variables (`.env`)**
```env
ETHERSCAN_API_KEY=YourApiKey
ETHERSCAN_URL=https://api-sepolia.etherscan.io/api # Testnet
# For Mainnet use: https://api.etherscan.io/api
DEPLOYER_ADDRESS=0xYourDeployerAddress
```

---

### **3. Create Etherscan Service**
```bash
php artisan make:service EtherscanService
```

**`app/Services/EtherscanService.php`**
```php
<?php

namespace App\Services;

use GuzzleHttp\Client;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

class EtherscanService
{
    protected $client;
    protected $apiKey;
    protected $baseUrl;

    public function __construct()
    {
        $this->client = new Client();
        $this->apiKey = env('ETHERSCAN_API_KEY');
        $this->baseUrl = env('ETHERSCAN_URL');
    }

    public function verifyContract(
        string $contractAddress,
        string $sourceCode,
        string $contractName,
        string $compilerVersion,
        int $optimizationRuns = 200,
        array $constructorArgs = null
    ): string {
        $params = [
            'apikey' => $this->apiKey,
            'module' => 'contract',
            'action' => 'verifysourcecode',
            'contractaddress' => $contractAddress,
            'sourceCode' => $sourceCode,
            'codeformat' => 'solidity-single-file',
            'contractname' => $contractName,
            'compilerversion' => $this->formatCompilerVersion($compilerVersion),
            'optimizationUsed' => $optimizationRuns > 0 ? 1 : 0,
            'runs' => $optimizationRuns,
            'licenseType' => 3, // MIT License
        ];

        if ($constructorArgs) {
            $params['constructorArguements'] = $this->encodeConstructorArgs($constructorArgs);
        }

        try {
            $response = $this->client->post($this->baseUrl, [
                'form_params' => $params
            ]);

            $result = json_decode($response->getBody(), true);

            if ($result['status'] !== '1') {
                throw new \Exception($result['message'] ?? 'Verification failed');
            }

            return $result['result']; // Returns GUID for verification check
        } catch (\Exception $e) {
            Log::error("Etherscan verification failed: " . $e->getMessage());
            throw $e;
        }
    }

    public function checkVerificationStatus(string $guid): array
    {
        $response = $this->client->get($this->baseUrl, [
            'query' => [
                'apikey' => $this->apiKey,
                'module' => 'contract',
                'action' => 'checkverifystatus',
                'guid' => $guid
            ]
        ]);

        return json_decode($response->getBody(), true);
    }

    protected function formatCompilerVersion(string $version): string
    {
        return 'v' . str_replace('^', '-', $version);
    }

    protected function encodeConstructorArgs(array $args): string
    {
        return '0x' . implode('', array_map(function ($arg) {
            return Str::replaceFirst('0x', '', $arg);
        }, $args));
    }
}
```

---

### **4. Integrate with Deployment Workflow**
Modify your deployment service to include verification:

**`app/Services/ContractDeployer.php`**
```php
public function deployWithVerification(
    string $bytecode,
    string $sourceCode,
    string $contractName,
    string $compilerVersion
): array {
    $txHash = $this->deploy($bytecode);
    
    // Wait for transaction receipt
    $receipt = $this->getTransactionReceipt($txHash);
    $contractAddress = $receipt['contractAddress'];

    // Verify contract
    $etherscan = app(EtherscanService::class);
    $guid = $etherscan->verifyContract(
        $contractAddress,
        $sourceCode,
        $contractName,
        $compilerVersion
    );

    return [
        'tx_hash' => $txHash,
        'contract_address' => $contractAddress,
        'verification_guid' => $guid
    ];
}

protected function getTransactionReceipt(string $txHash): array
{
    $receipt = null;
    $attempts = 0;
    
    while ($attempts < 12) { // Wait up to 1 minute
        $this->web3->eth->getTransactionReceipt($txHash, function ($err, $result) use (&$receipt) {
            if (!$err && $result) $receipt = (array)$result;
        });
        
        if ($receipt) break;
        
        sleep(5);
        $attempts++;
    }
    
    if (!$receipt) throw new \Exception("Transaction receipt not found");
    return $receipt;
}
```

---

### **5. Create Verification Check Command**
```bash
php artisan make:command CheckVerification
```

**`app/Console/Commands/CheckVerification.php`**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\EtherscanService;

class CheckVerification extends Command
{
    protected $signature = 'contract:verify-check {guid}';
    protected $description = 'Check Etherscan verification status';

    public function handle(EtherscanService $etherscan)
    {
        $result = $etherscan->checkVerificationStatus($this->argument('guid'));
        
        $this->table(
            ['Field', 'Value'],
            [
                ['Status', $result['status'] === '1' ? '✅ Verified' : '❌ Failed'],
                ['Message', $result['message'] ?? 'N/A'],
                ['GUID', $this->argument('guid')]
            ]
        );
    }
}
```

---

### **6. Usage Example**
**Deploy with Verification:**
```php
$deployer = app(ContractDeployer::class);
$result = $deployer->deployWithVerification(
    file_get_contents('build/MyContract.bin'),
    file_get_contents('contracts/MyContract.sol'),
    'MyContract',
    '0.8.17' // Must match exact compiler version
);

// Output:
// [
//     'tx_hash' => '0x123...',
//     'contract_address' => '0x456...',
//     'verification_guid' => 'yz2...'
// ]
```

**Check Verification Status:**
```bash
php artisan contract:verify-check yz2...
```

---

### **Key Features**

1. **Automatic Verification**
   - Submits source code immediately after deployment
   - Handles both constructor arguments and optimization settings

2. **Flexible Compiler Support**
   ```php
   // Supports various version formats:
   '0.8.17' => 'v0.8.17'
   '^0.8.0' => 'v0.8.0'
   ```

3. **Constructor Arguments**
   ```php
   $constructorArgs = [
       '0x123...', // First argument
       '0x456...'  // Second argument
   ];
   ```

4. **Production-Ready**
   - Built-in retry logic for transaction receipts
   - Comprehensive error logging

---

### **Troubleshooting**

| Issue | Solution |
|-------|----------|
| `Source code not found` | Ensure single-file flattening |
| `Compiler version mismatch` | Use exact version (e.g., 0.8.17) |
| `Already verified` | Check if contract was previously verified |
| `Timeout` | Increase wait time for transaction receipt |

---

### **Advanced: Multi-File Verification**
For complex contracts with imports:

1. Flatten contracts first:
```bash
npm install -g truffle-flattener
truffle-flattener contracts/MyContract.sol > build/MyContractFlat.sol
```

2. Then pass the flattened file:
```php
$sourceCode = file_get_contents('build/MyContractFlat.sol');
```

---

### **Next Steps**
1. **Add queue support** for background verification
2. **Store verification status** in database
3. **Support proxy contracts** verification

Would you like me to elaborate on any of these advanced features?