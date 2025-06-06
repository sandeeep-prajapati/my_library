### File Name: `15_build_laravel_dashboard_for_balances_transactions.md`

---

### **Step 1: Install Dependencies**
Install Laravel and required packages:

```bash
composer require web3p/web3.php
composer require laravel/ui
php artisan ui bootstrap --auth
npm install && npm run dev
```

---

### **Step 2: Create Models and Migrations**
Run these commands to create models and migrations for transactions, user balances, and contract events:

```bash
php artisan make:model Transaction -m
php artisan make:model UserBalance -m
php artisan make:model ContractEvent -m
```

**`database/migrations/create_transactions_table.php`**
```php
Schema::create('transactions', function (Blueprint $table) {
    $table->id();
    $table->unsignedBigInteger('user_id');
    $table->string('from');
    $table->string('to');
    $table->decimal('value', 18, 8);
    $table->string('transaction_hash')->unique();
    $table->timestamps();
});
```

**`database/migrations/create_user_balances_table.php`**
```php
Schema::create('user_balances', function (Blueprint $table) {
    $table->id();
    $table->unsignedBigInteger('user_id')->unique();
    $table->decimal('balance', 18, 8)->default(0);
    $table->timestamps();
});
```

**`database/migrations/create_contract_events_table.php`**
```php
Schema::create('contract_events', function (Blueprint $table) {
    $table->id();
    $table->string('event_name');
    $table->string('transaction_hash');
    $table->json('event_data');
    $table->timestamps();
});
```

Run the migrations:

```bash
php artisan migrate
```

---

### **Step 3: Create Controllers**
Run these commands to create controllers for each entity:

```bash
php artisan make:controller DashboardController
php artisan make:controller TransactionController
php artisan make:controller EventController
```

---

### **Step 4: Implement Controller Logic**
**`DashboardController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\UserBalance;
use App\Models\Transaction;
use App\Models\ContractEvent;

class DashboardController extends Controller
{
    public function index()
    {
        $balances = UserBalance::all();
        $transactions = Transaction::latest()->limit(10)->get();
        $events = ContractEvent::latest()->limit(10)->get();

        return view('dashboard', compact('balances', 'transactions', 'events'));
    }
}
```

**`TransactionController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Transaction;

class TransactionController extends Controller
{
    public function index()
    {
        $transactions = Transaction::latest()->paginate(10);
        return view('transactions.index', compact('transactions'));
    }
}
```

**`EventController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\ContractEvent;

class EventController extends Controller
{
    public function index()
    {
        $events = ContractEvent::latest()->paginate(10);
        return view('events.index', compact('events'));
    }
}
```

---

### **Step 5: Create Blade Views**
**`resources/views/dashboard.blade.php`**
```blade
@extends('layouts.app')

@section('content')
<div class="container">
    <h1 class="mb-4">Dashboard</h1>

    <h3>User Balances</h3>
    <table class="table table-bordered">
        <tr>
            <th>User ID</th>
            <th>Balance (ETH)</th>
        </tr>
        @foreach($balances as $balance)
        <tr>
            <td>{{ $balance->user_id }}</td>
            <td>{{ $balance->balance }}</td>
        </tr>
        @endforeach
    </table>

    <h3>Recent Transactions</h3>
    <table class="table table-bordered">
        <tr>
            <th>From</th>
            <th>To</th>
            <th>Value (ETH)</th>
            <th>Hash</th>
        </tr>
        @foreach($transactions as $tx)
        <tr>
            <td>{{ $tx->from }}</td>
            <td>{{ $tx->to }}</td>
            <td>{{ $tx->value }}</td>
            <td>{{ $tx->transaction_hash }}</td>
        </tr>
        @endforeach
    </table>

    <h3>Recent Events</h3>
    <table class="table table-bordered">
        <tr>
            <th>Event Name</th>
            <th>Transaction Hash</th>
            <th>Data</th>
        </tr>
        @foreach($events as $event)
        <tr>
            <td>{{ $event->event_name }}</td>
            <td>{{ $event->transaction_hash }}</td>
            <td>{{ json_encode($event->event_data) }}</td>
        </tr>
        @endforeach
    </table>
</div>
@endsection
```

---

### **Step 6: Define Routes**
In `routes/web.php`:

```php
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\TransactionController;
use App\Http\Controllers\EventController;

Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard');
Route::get('/transactions', [TransactionController::class, 'index'])->name('transactions.index');
Route::get('/events', [EventController::class, 'index'])->name('events.index');
```

---

### **Step 7: Add Web3 Integration for Data Fetching**
Add logic to fetch blockchain data via Web3 PHP in your controllers or commands.

Example Web3 PHP logic for fetching balances:

```php
use Web3\Web3;

$web3 = new Web3('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID');
$contract = new Contract($web3->provider, 'YOUR_ERC20_ABI');
$address = '0xYourWalletAddress';

$contract->at('0xYourContractAddress')->call('balanceOf', $address, function ($err, $result) {
    if ($err) {
        return "Error: " . $err->getMessage();
    }
    return "Balance: " . hexdec($result[0]) / 1e18 . " ETH";
});
```

---

### **Step 8: Test the Dashboard**
1. Start Laravel server:  
   ```bash
   php artisan serve
   ```

2. Visit the dashboard URL:  
   ```
   http://localhost:8000/dashboard
   ```

---

### **Step 9: Bonus Features for Improvement**
✅ Add filters for viewing transactions by date, amount, etc.  
✅ Include charts using libraries like `Chart.js` or `ApexCharts`.  
✅ Implement real-time updates using Laravel Echo or WebSockets.  
✅ Add role-based access control to restrict admin functionality.  

If you'd like detailed steps for any enhancements, let me know! 🚀