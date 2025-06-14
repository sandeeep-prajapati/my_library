Here's a complete guide to creating a Laravel Artisan command to check ETH/BNB balances:

### 1. Create the Command
```bash
php artisan make:command CheckCryptoBalance
```

### 2. Implement the Command Logic
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Web3\Web3;
use Web3\Providers\HttpProvider;
use Web3\RequestManagers\HttpRequestManager;

class CheckCryptoBalance extends Command
{
    protected $signature = 'crypto:balance 
                            {address : Wallet address (0x...)}
                            {--chain=eth : Blockchain (eth or bsc)}
                            {--decimals=18 : Token decimals}';
    
    protected $description = 'Check ETH/BNB balance of any address';

    public function handle()
    {
        $address = $this->argument('address');
        $chain = $this->option('chain');
        $decimals = (int)$this->option('decimals');

        // Validate address
        if (!preg_match('/^0x[a-fA-F0-9]{40}$/', $address)) {
            $this->error('Invalid wallet address format');
            return 1;
        }

        try {
            $balance = $this->getBalance($address, $chain);
            $formatted = $this->formatBalance($balance, $decimals);

            $this->line('');
            $this->info("$chain Balance for $address");
            $this->line(str_repeat('-', 50));
            $this->line("Wei: $balance");
            $this->line("Normalized: $formatted");
            $this->line('');

            return 0;
        } catch (\Exception $e) {
            $this->error("Error: " . $e->getMessage());
            return 1;
        }
    }

    protected function getBalance(string $address, string $chain): string
    {
        $rpcUrl = $this->getRpcUrl($chain);
        $web3 = new Web3(new HttpProvider(new HttpRequestManager($rpcUrl, 10)));

        $balance = '';
        $web3->eth->getBalance($address, function ($err, $result) use (&$balance) {
            if ($err) {
                throw new \Exception($err->getMessage());
            }
            $balance = $result->toString();
        });

        return $balance;
    }

    protected function formatBalance(string $wei, int $decimals): string
    {
        $value = bcdiv($wei, bcpow('10', $decimals), $decimals);
        return number_format((float)$value, 6, '.', '');
    }

    protected function getRpcUrl(string $chain): string
    {
        return match(strtolower($chain)) {
            'eth' => config('blockchain.ethereum.rpc_url'),
            'bsc' => config('blockchain.bsc.rpc_url'),
            default => throw new \Exception("Unsupported chain: $chain")
        };
    }
}
```

### 3. Configure RPC URLs
Add to your `config/blockchain.php`:
```php
return [
    'ethereum' => [
        'rpc_url' => env('ETH_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_KEY'),
    ],
    'bsc' => [
        'rpc_url' => env('BSC_RPC_URL', 'https://bsc-dataseed.binance.org/'),
    ],
];
```

### 4. Usage Examples

Check ETH balance:
```bash
php artisan crypto:balance 0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

Check BNB balance:
```bash
php artisan crypto:balance 0x742d35Cc6634C0532925a3b844Bc454e4438f44e --chain=bsc
```

With custom decimals (for tokens):
```bash
php artisan crypto:balance 0x742d35Cc6634C0532925a3b844Bc454e4438f44e --decimals=6
```

### 5. Sample Output
```
ETH Balance for 0x742d35Cc6634C0532925a3b844Bc454e4438f44e
--------------------------------------------------
Wei: 3492015000000000000
Normalized: 3.492015
```

### Key Features:

1. **Multi-chain Support**: Handles both Ethereum and BSC
2. **Flexible Decimals**: Supports different token decimals
3. **Validation**: Checks address format
4. **Error Handling**: Catches RPC errors gracefully
5. **Human-readable**: Formats balance in both wei and ETH/BNB
6. **Configurable**: RPC endpoints in config file

### Error Handling Cases:

- Invalid address format
- Unsupported blockchain
- RPC connection issues
- Timeout handling (10 seconds)

To extend this for ERC20 tokens, you would:
1. Add a `--token` option
2. Implement ABI contract calls
3. Use the token's contract address

Would you like me to add the ERC20 token balance checking functionality as well?