### File Name: `12_listen_to_erc20_transfer_events.md`

---

### **Step 1: Install Required Libraries**
To listen to ERC20 contract events in PHP (Laravel), you need the following libraries:  
- `web3p/web3.php` (for Ethereum interaction)
- `laravel/framework` (if using Laravel)

Install them via Composer:  
```bash
composer require web3p/web3.php
```

---

### **Step 2: Create a Laravel Command**
Create a custom command for listening to ERC20 events.  

Run this command:  
```bash
php artisan make:command ListenToERC20Events
```

---

### **Step 3: Write the PHP Script**
Inside your generated command file (`app/Console/Commands/ListenToERC20Events.php`), write the logic for listening to ERC20 `Transfer` events.

**Example Code:**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Web3\Web3;
use Web3\Contract;

class ListenToERC20Events extends Command
{
    protected $signature = 'listen:erc20';
    protected $description = 'Listen for ERC20 Transfer events';

    public function handle()
    {
        $web3 = new Web3('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID');
        $contract = new Contract($web3->provider, 'YOUR_ERC20_ABI');

        $contractAddress = '0xYourContractAddress';

        // Subscribe to 'Transfer' events
        $contract->at($contractAddress)->getEvent('Transfer', function ($err, $event) {
            if ($err !== null) {
                $this->error("Error: " . $err->getMessage());
                return;
            }

            $this->info("Transfer Event Detected!");
            $this->info("From: " . $event['args']['from']);
            $this->info("To: " . $event['args']['to']);
            $this->info("Value: " . $event['args']['value']);

            // Log data to a file
            \Log::info('ERC20 Transfer Event:', [
                'from' => $event['args']['from'],
                'to' => $event['args']['to'],
                'value' => $event['args']['value']
            ]);
        });
    }
}
```

---

### **Step 4: Add ABI for ERC20 Contract**
Your ABI (Application Binary Interface) should match the ERC20 contract’s structure. Example ERC20 ABI snippet for the `Transfer` event:

```json
[
    {
        "anonymous": false,
        "inputs": [
            { "indexed": true, "name": "from", "type": "address" },
            { "indexed": true, "name": "to", "type": "address" },
            { "indexed": false, "name": "value", "type": "uint256" }
        ],
        "name": "Transfer",
        "type": "event"
    }
]
```

---

### **Step 5: Schedule the Command**
In `app/Console/Kernel.php`, schedule the command:

```php
protected function schedule(Schedule $schedule)
{
    $schedule->command('listen:erc20')->everyMinute();
}
```

Then run the scheduler with:  
```bash
php artisan schedule:work
```

---

### **Step 6: Run the Command Manually (For Testing)**
To test the listener manually, run:  
```bash
php artisan listen:erc20
```

---

### **Step 7: Logs & Verification**
- Successful events will appear in the terminal.
- Logged data will be stored in Laravel’s log file:  
`storage/logs/laravel.log`

---

### **Bonus Tip:**  
If you want real-time event detection with WebSockets or a queue system, consider integrating Laravel’s broadcasting system or services like Pusher.

Let me know if you'd like further explanations or additional features! 🚀