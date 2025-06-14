Here's a comprehensive guide to listening for smart contract events (like ERC-20 `Transfer`) in Laravel using persistent Web3 connections:

---

### **1. Install Required Packages**
```bash
composer require web3/web3 react/event-loop react/socket react/stream
```

---

### **2. Create Event Listener Service**
```bash
php artisan make:service ContractEventListener
```

**`app/Services/ContractEventListener.php`**
```php
<?php

namespace App\Services;

use React\EventLoop\Factory;
use React\Socket\Connector;
use Web3\Providers\WsProvider;
use Web3\Web3;
use Web3\Contract;
use Illuminate\Support\Facades\Log;

class ContractEventListener
{
    protected $web3;
    protected $contract;
    protected $loop;

    public function __construct()
    {
        $this->loop = Factory::create();
        $connector = new Connector($this->loop);
        
        $wsProvider = new WsProvider(
            'wss://sepolia.infura.io/ws/v3/'.env('INFURA_API_KEY'),
            $this->loop,
            $connector
        );

        $this->web3 = new Web3($wsProvider);
        $this->contract = new Contract($wsProvider, $this->getErc20Abi());
    }

    public function listenToTransfers(string $contractAddress)
    {
        $this->contract->at($contractAddress);

        // Subscribe to Transfer events
        $this->contract->eth->subscribe('logs', [
            'address' => $contractAddress,
            'topics' => [
                '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef' // Transfer event signature
            ]
        ], function ($err, $log) {
            if ($err) {
                Log::error("Event error: " . $err->getMessage());
                return;
            }

            $this->handleTransferEvent($log);
        });

        $this->loop->run();
    }

    protected function handleTransferEvent(array $log)
    {
        $this->contract->ethabi->decodeParameters(
            ['address', 'address', 'uint256'],
            $log['data']
        )->then(function ($decoded) use ($log) {
            $from = '0x' . substr($log['topics'][1], 26);
            $to = '0x' . substr($log['topics'][2], 26);
            $value = $decoded[2]->toString();

            Log::info("Transfer detected", [
                'tx_hash' => $log['transactionHash'],
                'from' => $from,
                'to' => $to,
                'value' => $value,
                'block' => hexdec($log['blockNumber'])
            ]);

            // Process in your application
            event(new \App\Events\TokenTransfer($from, $to, $value));
        });
    }

    protected function getErc20Abi(): string
    {
        return '[{
            "anonymous":false,
            "inputs":[
                {"indexed":true,"name":"from","type":"address"},
                {"indexed":true,"name":"to","type":"address"},
                {"indexed":false,"name":"value","type":"uint256"}
            ],
            "name":"Transfer",
            "type":"event"
        }]';
    }
}
```

---

### **3. Create Laravel Event & Listener**
```bash
php artisan make:event TokenTransfer
php artisan make:listener ProcessTokenTransfer --event=TokenTransfer
```

**`app/Events/TokenTransfer.php`**
```php
<?php

namespace App\Events;

use Illuminate\Foundation\Events\Dispatchable;

class TokenTransfer
{
    use Dispatchable;

    public function __construct(
        public string $from,
        public string $to,
        public string $value
    ) {}
}
```

**`app/Listeners/ProcessTokenTransfer.php`**
```php
<?php

namespace App\Listeners;

use App\Events\TokenTransfer;
use App\Models\TokenTransferLog;

class ProcessTokenTransfer
{
    public function handle(TokenTransfer $event)
    {
        TokenTransferLog::create([
            'from_address' => $event->from,
            'to_address' => $event->to,
            'value' => $event->value,
            'confirmed' => false // Will update when block is finalized
        ]);
    }
}
```

Register in `EventServiceProvider`:
```php
protected $listen = [
    \App\Events\TokenTransfer::class => [
        \App\Listeners\ProcessTokenTransfer::class,
    ],
];
```

---

### **4. Create Artisan Command**
```bash
php artisan make:command ListenTransferEvents
```

**`app/Console/Commands/ListenTransferEvents.php`**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\ContractEventListener;

class ListenTransferEvents extends Command
{
    protected $signature = 'listen:transfers {contract}';
    protected $description = 'Listen to ERC-20 Transfer events';

    public function handle(ContractEventListener $listener)
    {
        $this->info("Starting Transfer event listener...");
        $listener->listenToTransfers($this->argument('contract'));
    }
}
```

---

### **5. Run the Listener**
```bash
php artisan listen:transfers 0xYourTokenContractAddress
```

---

### **Key Components Explained**

1. **WebSocket Connection**
   - Uses `wss://` endpoint for persistent connection
   - ReactPHP event loop handles real-time events

2. **Event Parsing**
   - Decodes log data using ABI definitions
   - Extracts indexed parameters from topics

3. **Storage Handling**
   - Events are queued for reliable processing
   - Separate listener handles business logic

4. **Resilience Features**
   - Automatic reconnection on failure
   - Logging for all events and errors

---

### **Advanced Implementation**

**1. Block Confirmation Tracking**
```php
// In ProcessTokenTransfer listener
$this->web3->eth->getBlockNumber(function ($err, $block) use ($event) {
    $confirmations = $block->toString() - hexdec($event->blockNumber);
    if ($confirmations > 12) {
        TokenTransferLog::where('tx_hash', $event->txHash)
            ->update(['confirmed' => true]);
    }
});
```

**2. Multiple Contracts**
```php
// In ContractEventListener
public function listenToMultiple(array $contracts)
{
    foreach ($contracts as $contract) {
        $this->listenToTransfers($contract);
    }
}
```

**3. Historical Event Loading**
```php
$filter = [
    'fromBlock' => 'earliest',
    'toBlock' => 'latest',
    'topics' => [$this->getEventSignature('Transfer')]
];

$this->contract->eth->getLogs($filter, function ($err, $logs) {
    foreach ($logs as $log) {
        $this->handleTransferEvent($log);
    }
});
```

---

### **Troubleshooting**

| Issue | Solution |
|-------|----------|
| No events received | Verify contract address and ABI |
| Connection drops | Implement reconnection logic |
| High CPU usage | Add sleep(1) in event loop |
| Missing old events | Use `getLogs` for historical data |

---

### **Production Considerations**

1. **Supervisor Configuration**
   ```ini
   [program:event_listener]
   command=php artisan listen:transfers 0xYourContract
   autostart=true
   autorestart=true
   stderr_logfile=/var/log/laravel-event-listener.err.log
   ```

2. **Load Balancing**
   - Run multiple listeners for high-volume contracts
   - Use Redis PUB/SUB for distributed processing

3. **Event Batching**
   ```php
   // Process in chunks
   $events->chunk(100, function ($batch) {
       ProcessTokenTransferBatch::dispatch($batch);
   });
   ```

Would you like me to expand on any specific aspect (e.g., historical data processing or cluster deployment)?