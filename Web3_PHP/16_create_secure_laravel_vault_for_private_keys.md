### File Name: `16_create_secure_laravel_vault_for_private_keys.md`

---

## **Step 1: Install Laravel and Required Packages**
If Laravel is not already installed, run:

```bash
composer create-project laravel/laravel secure-vault
```

Next, install the Web3 PHP package:

```bash
composer require web3p/web3.php
```

---

## **Step 2: Configure Laravel Encryption**
Laravel provides a secure way to encrypt and decrypt sensitive data using `Crypt` or `Encryptable` traits.

In your `.env` file, ensure you have the `APP_KEY` set:

```env
APP_KEY=base64:YOUR_APP_KEY
```

If it's missing, generate it using:

```bash
php artisan key:generate
```

---

## **Step 3: Create a Vault Model and Migration**
Create a model with a migration:

```bash
php artisan make:model Vault -m
```

**`database/migrations/create_vaults_table.php`**
```php
Schema::create('vaults', function (Blueprint $table) {
    $table->id();
    $table->unsignedBigInteger('user_id')->unique();
    $table->text('encrypted_private_key');
    $table->timestamps();
});
```

Run the migration:

```bash
php artisan migrate
```

---

## **Step 4: Create the Vault Controller**
Create a controller to handle encryption and decryption logic:

```bash
php artisan make:controller VaultController
```

---

## **Step 5: Implement Encryption and Decryption Logic**
**`VaultController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Vault;
use Illuminate\Support\Facades\Crypt;

class VaultController extends Controller
{
    // Store Private Key
    public function store(Request $request)
    {
        $request->validate([
            'user_id' => 'required|exists:users,id',
            'private_key' => 'required|string'
        ]);

        $encryptedKey = Crypt::encryptString($request->private_key);

        Vault::updateOrCreate(
            ['user_id' => $request->user_id],
            ['encrypted_private_key' => $encryptedKey]
        );

        return response()->json(['message' => 'Private key stored securely.']);
    }

    // Decrypt Private Key for Signing Transactions
    public function getDecryptedKey($userId)
    {
        $vault = Vault::where('user_id', $userId)->firstOrFail();
        $decryptedKey = Crypt::decryptString($vault->encrypted_private_key);

        return response()->json(['private_key' => $decryptedKey]);
    }
}
```

---

## **Step 6: Define Routes**
In `routes/web.php`:

```php
use App\Http\Controllers\VaultController;

Route::post('/vault/store', [VaultController::class, 'store']);
Route::get('/vault/decrypt/{userId}', [VaultController::class, 'getDecryptedKey']);
```

---

## **Step 7: Secure Environment Setup**
To enhance security:

✅ Add `.env` values for Infura/Alchemy URLs  
✅ Ensure `.env` file permissions are restricted (`chmod 600 .env`)  
✅ Use environment variables for contract addresses and sensitive data  

Example `.env` setup:
```env
INFURA_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
PRIVATE_KEY=encrypted_data_here
```

---

## **Step 8: Implement Secure Signing with Web3 PHP**
In a Laravel command or controller:

**`SignTransactionController.php`**
```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Web3\Web3;
use Illuminate\Support\Facades\Crypt;
use App\Models\Vault;

class SignTransactionController extends Controller
{
    public function signTransaction(Request $request)
    {
        $vault = Vault::where('user_id', $request->user_id)->firstOrFail();
        $privateKey = Crypt::decryptString($vault->encrypted_private_key);

        $web3 = new Web3(env('INFURA_URL'));
        $account = $web3->eth->accounts->privateKeyToAccount($privateKey);

        $transaction = [
            'from' => $account->address,
            'to' => $request->to,
            'value' => '0x' . dechex($request->value * 10 ** 18),
            'gas' => '0x5208', // 21000
        ];

        $signedTx = $account->signTransaction($transaction);

        return response()->json(['signed_transaction' => $signedTx]);
    }
}
```

---

## **Step 9: Add Middleware for Extra Security**
Protect the vault routes with authentication and role-based checks:

**`app/Http/Middleware/VerifyAdmin.php`**
```php
<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class VerifyAdmin
{
    public function handle(Request $request, Closure $next)
    {
        if (!auth()->check() || auth()->user()->role !== 'admin') {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        return $next($request);
    }
}
```

Add the middleware to your route:

```php
Route::middleware(['auth', 'verifyAdmin'])->group(function () {
    Route::post('/vault/store', [VaultController::class, 'store']);
});
```

---

## **Step 10: Testing**
✅ **Unit Test:** Test encryption, decryption, and data storage.  
✅ **Transaction Test:** Verify transaction signing with sample data.  
✅ **Security Test:** Check unauthorized access attempts.

---

## **Step 11: Best Practices for Extra Security**
✅ Avoid storing plaintext private keys in `.env`.  
✅ Use Laravel’s native encryption instead of custom solutions.  
✅ Implement IP restrictions for critical endpoints.  
✅ Consider using AWS KMS, Azure Key Vault, or HashiCorp Vault for enterprise-grade security.  

---

## **Step 12: Sample API Testing**
Using Postman or cURL:

### **Store Private Key**
```bash
curl -X POST -d "user_id=1&private_key=0xYOUR_PRIVATE_KEY" http://localhost:8000/vault/store
```

### **Retrieve Decrypted Key (For Testing)**
```bash
curl -X GET http://localhost:8000/vault/decrypt/1
```

### **Sign Transaction**
```bash
curl -X POST -d "user_id=1&to=0xRecipientAddress&value=0.01" http://localhost:8000/sign-transaction
```

---

If you'd like further enhancements like key rotation, multi-signature support, or real-time alerts for key access, let me know! 🚀