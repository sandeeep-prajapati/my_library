### File Name: `14_fetch_token_balances_command.md`

---

### **Step 1: Install Dependencies**
Install the required package for interacting with the Ethereum blockchain:  

```bash
composer require web3p/web3.php
```

---

### **Step 2: Create a Laravel Command**
Run the following command to generate a custom command:  

```bash
php artisan make:command FetchTokenBalances
```

---

### **Step 3: Create the `user_balances` Table**
Run this command to create a model and migration:

```bash
php artisan make:model UserBalance -m
```

In the generated migration file (`database/migrations/xxxx_xx_xx_create_user_balances_table.php`), define the table structure:

```php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateUserBalancesTable extends Migration
{
    public function up()
    {
        Schema::create('user_balances', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id');
            $table->decimal('balance', 18, 8)->default(0);
            $table->timestamps();

            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade');
        });
    }

    public function down()
    {
        Schema::dropIfExists('user_balances');
    }
}
```

Run the migration with:  
```bash
php artisan migrate
```

---

### **Step 4: Write Command Logic**
In `app/Console/Commands/FetchTokenBalances.php`, implement the logic to fetch token balances.

**Example Code:**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Web3\Web3;
use Web3\Contract;
use App\Models\User;
use App\Models\UserBalance;

class FetchTokenBalances extends Command
{
    protected $signature = 'fetch:token-balances';
    protected $description = 'Fetch token balances for all users and update the user_balances table';

    public function handle()
    {
        $web3 = new Web3('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID');
        $contract = new Contract($web3->provider, 'YOUR_ERC20_ABI');

        $contractAddress = '0xYourContractAddress';
        
        $users = User::all();

        foreach ($users as $user) {
            $contract->at($contractAddress)->call('balanceOf', $user->wallet_address, function ($err, $result) use ($user) {
                if ($err !== null) {
                    $this->error("Error fetching balance for user {$user->id}: " . $err->getMessage());
                    return;
                }

                $balance = hexdec($result[0]);

                // Update or create user balance
                UserBalance::updateOrCreate(
                    ['user_id' => $user->id],
                    ['balance' => $balance / 1e18]  // Convert from Wei to Ether
                );

                $this->info("Updated balance for user {$user->id}: " . $balance / 1e18);
            });
        }

        $this->info('Token balances updated successfully!');
    }
}
```

---

### **Step 5: Schedule the Command**
In `app/Console/Kernel.php`, schedule the command to run every hour:

```php
protected function schedule(Schedule $schedule)
{
    $schedule->command('fetch:token-balances')->hourly();
}
```

Run the scheduler with:  
```bash
php artisan schedule:work
```

---

### **Step 6: Run the Command Manually (For Testing)**
To test the command manually, run:  
```bash
php artisan fetch:token-balances
```

---

### **Step 7: Verify Results**
✅ Check your `user_balances` table for updated balances.  
✅ Check `storage/logs/laravel.log` for error details (if any).

---

### **Bonus Tips for Improvement**
✅ Add error handling to retry failed API calls.  
✅ Implement logging for debugging and tracking balance updates.  
✅ Consider using queues for improved scalability if you have a large user base.  

If you'd like guidance on enhancing this system further, let me know! 🚀