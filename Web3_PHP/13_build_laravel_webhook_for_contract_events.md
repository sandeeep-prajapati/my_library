    ### File Name: `13_build_laravel_webhook_for_contract_events.md`

---

### **Step 1: Install Dependencies**
To build a webhook listener in Laravel for Ethereum contract events, you'll need:

- **Infura** or **Alchemy** for blockchain interaction
- `web3p/web3.php` for Ethereum JSON-RPC calls

Install the required package:  
```bash
composer require web3p/web3.php
```

---

### **Step 2: Create a Webhook Controller**
Run this command to generate a controller for webhook handling:  
```bash
php artisan make:controller WebhookController
```

---

### **Step 3: Define Webhook Logic**
In `app/Http/Controllers/WebhookController.php`, write the logic to handle incoming event data and update your database.

**Example Code:**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Transaction;
use Web3\Web3;
use Web3\Contract;

class WebhookController extends Controller
{
    public function handleEvent(Request $request)
    {
        // Validate incoming data
        $request->validate([
            'transactionHash' => 'required|string',
            'event' => 'required|array',
        ]);

        $web3 = new Web3('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID');
        $contract = new Contract($web3->provider, 'YOUR_ERC20_ABI');
        
        $contractAddress = '0xYourContractAddress';

        // Extract event details
        $eventData = $request->input('event');

        if (isset($eventData['args'])) {
            Transaction::create([
                'from' => $eventData['args']['from'],
                'to' => $eventData['args']['to'],
                'value' => hexdec($eventData['args']['value']),
                'transaction_hash' => $request->input('transactionHash'),
            ]);

            return response()->json(['message' => 'Event processed successfully'], 200);
        }

        return response()->json(['error' => 'Invalid event data'], 400);
    }
}
```

---

### **Step 4: Create a Transaction Model and Migration**
Run the following command to create a model with a migration:

```bash
php artisan make:model Transaction -m
```

In the generated migration file (`database/migrations/xxxx_xx_xx_create_transactions_table.php`), define the database structure:

```php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateTransactionsTable extends Migration
{
    public function up()
    {
        Schema::create('transactions', function (Blueprint $table) {
            $table->id();
            $table->string('from');
            $table->string('to');
            $table->decimal('value', 18, 8);
            $table->string('transaction_hash')->unique();
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('transactions');
    }
}
```

Run the migration with:  
```bash
php artisan migrate
```

---

### **Step 5: Define Webhook Route**
In `routes/api.php`, add the webhook route:

```php
use App\Http\Controllers\WebhookController;

Route::post('/webhook/event', [WebhookController::class, 'handleEvent']);
```

---

### **Step 6: Register Webhook in Infura/Alchemy**
1. Log in to your **Infura** or **Alchemy** dashboard.
2. Navigate to your project and select **Webhooks**.
3. Add your webhook URL as:  
```
https://yourdomain.com/api/webhook/event
```

---

### **Step 7: Test the Webhook**
For local testing, you can use `ngrok` to expose your local server:

1. Start your Laravel server:  
```bash
php artisan serve
```

2. Run ngrok to expose your app:  
```bash
ngrok http 8000
```

3. Use the generated ngrok URL (e.g., `https://abcd1234.ngrok.io`) in your Infura/Alchemy webhook settings.

---

### **Step 8: Verify Incoming Data**
- Check `storage/logs/laravel.log` for incoming event data.
- Ensure the database is updated with the transaction details.

---

### **Bonus Tip:**  
For improved security:
✅ Use Laravel middleware to authenticate webhook requests.  
✅ Add rate-limiting rules to protect your webhook endpoint.

If you’d like an example of securing the webhook or enhancing the database design, let me know! 🚀