
---

# 📄 01_setup.md  
## 📌 Title: Installing Web3 PHP in Laravel & Connecting to Ethereum Mainnet via Infura

---

## 🔗 Step 1: Install Web3 PHP Package

Laravel doesn’t have native Web3 support, so we’ll use a great library called `web3-php`.

Run this command to install it via Composer:
```bash
composer require web3/web3
```

---

## 🔗 Step 2: Set up Infura Account
1. Go to [https://infura.io](https://infura.io) and create an account.
2. Create a new project (e.g., `LaravelWeb3Project`).
3. In the project dashboard, select the **Ethereum** network and choose **Mainnet**.
4. Copy the **Infura Project ID** (we’ll need this to connect).

---

## 🔗 Step 3: Add Configuration to `.env`

In your `.env` file, add:

```env
ETHEREUM_NETWORK=mainnet
INFURA_PROJECT_ID=your_infura_project_id_here
INFURA_PROJECT_SECRET=your_infura_project_secret_here  # (optional if required)
```

---

## 🔗 Step 4: Create a Laravel Config File for Web3 (optional but recommended)

Create a new config file: `config/web3.php`

```php
return [
    'network' => env('ETHEREUM_NETWORK', 'mainnet'),
    'infura_project_id' => env('INFURA_PROJECT_ID'),
    'infura_project_secret' => env('INFURA_PROJECT_SECRET', null), // For public endpoints, this may be optional
    'rpc_url' => env('INFURA_RPC_URL', 'https://mainnet.infura.io/v3/' . env('INFURA_PROJECT_ID')),
];
```

---

## 🔗 Step 5: Create a Web3 Service Class

Create a new service file in your Laravel project:
`app/Services/Web3Service.php`

```php
<?php

namespace App\Services;

use Web3\Web3;

class Web3Service
{
    protected $web3;

    public function __construct()
    {
        $rpcUrl = config('web3.rpc_url');
        $this->web3 = new Web3($rpcUrl);
    }

    public function getBlockNumber()
    {
        $blockNumber = null;

        $this->web3->eth->blockNumber(function ($err, $block) use (&$blockNumber) {
            if ($err !== null) {
                throw new \Exception("Error fetching block number: " . $err->getMessage());
            }
            $blockNumber = $block->toString();
        });

        return $blockNumber;
    }
}
```

---

## 🔗 Step 6: Create a Controller to Test Connection

Create: `app/Http/Controllers/Web3Controller.php`

```php
<?php

namespace App\Http\Controllers;

use App\Services\Web3Service;

class Web3Controller extends Controller
{
    protected $web3Service;

    public function __construct(Web3Service $web3Service)
    {
        $this->web3Service = $web3Service;
    }

    public function index()
    {
        try {
            $blockNumber = $this->web3Service->getBlockNumber();
            return response()->json(['block_number' => $blockNumber]);
        } catch (\Exception $e) {
            return response()->json(['error' => $e->getMessage()], 500);
        }
    }
}
```

---

## 🔗 Step 7: Add Route to Test It

In `routes/web.php` add:

```php
use App\Http\Controllers\Web3Controller;

Route::get('/web3/block-number', [Web3Controller::class, 'index']);
```

---

## 🔗 Step 8: Test Connection
Run your Laravel project and visit:
```
http://localhost:8000/web3/block-number
```

✅ If everything is correct, you will see:
```json
{
    "block_number": "18482984"
}
```

---

## 💡 Final Folder Structure Example
```
app/
├── Http/
│   ├── Controllers/
│   │   ├── Web3Controller.php
├── Services/
│   ├── Web3Service.php
config/
├── web3.php
routes/
├── web.php
.env
```

---

## ✅ Summary
| Step | Action |
|---|---|
| 1 | Install web3-php via Composer |
| 2 | Get Infura credentials |
| 3 | Add config in `.env` and `config/web3.php` |
| 4 | Create `Web3Service.php` to handle connection |
| 5 | Create `Web3Controller.php` to test |
| 6 | Add route and test connection |

---

## 🎁 Example .env
```env
ETHEREUM_NETWORK=mainnet
INFURA_PROJECT_ID=abc1234567890abcdef
INFURA_PROJECT_SECRET=optional_secret_here
```

---

## 📚 References
- [web3-php GitHub](https://github.com/web3p/web3.php)
- [Infura Documentation](https://infura.io/docs)

---
