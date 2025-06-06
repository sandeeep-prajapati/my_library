
---

## 🔥 20 Web3 PHP Mastery Prompts

### 📚 Basic Setup & Initialization
1. **Install Web3 PHP in Laravel and initialize a connection to Ethereum Mainnet via Infura. Share your complete setup and configuration files.**
2. **Write a service class in Laravel to connect to a local Ethereum node using Hardhat. Explain the difference between Infura and Hardhat.**

### 📦 Blockchain Fundamentals
3. **Write a PHP script (using Web3 PHP) to fetch and display the latest Ethereum block number.**
4. **Create a Laravel command that fetches the latest block number every minute and logs it to a file.**

### 💰 Wallet Operations
5. **Using Web3 PHP, create a function that retrieves the ETH balance of a given address. Build a Laravel API that accepts an address and returns the balance.**
6. **Create a function to generate a new Ethereum address (with private key) using Web3 PHP. Securely store it in Laravel's database.**

### 🔗 Smart Contract Interaction
7. **Write a function to read a public variable (like `name` or `symbol`) from an ERC20 contract using Web3 PHP.**
8. **Write a PHP script to call a "view" function from your custom smart contract (like `getTotalSupply`).**
9. **Create a Laravel endpoint to trigger a write transaction (like `transfer` in an ERC20 contract).**

### 🔥 Gas Estimation & Transaction Management
10. **Create a gas estimator function in Web3 PHP that calculates the gas required for sending 0.1 ETH.**



### 🔍 Event Listening & Handling
12. **Write a PHP script to listen for a specific event (like `Transfer`) from an ERC20 contract. Process and log the event data in Laravel.**
13. **Build a Laravel webhook that listens to Infura/Alchemy and updates your database when a smart contract emits an event.**

### 🗂️ Off-chain + On-chain Sync
14. **Create a Laravel command that fetches the token balance of all users (stored in your database) every hour and updates a `user_balances` table.**
15. **Build a complete Laravel dashboard to display user balances, recent transactions, and contract events (fetched using Web3 PHP).**

### 🔑 Private Key Management (Carefully!)
16. **Create a secure Laravel Vault to store private keys. Use Laravel Encryption to keep them safe and only decrypt when signing transactions.**
17. **Write a Laravel service class to sign and broadcast a raw transaction using a stored private key (using Web3 PHP).**

### 🌐 Multi-Network Support
18. **Extend your Laravel app to support switching between Ethereum Mainnet, Goerli, Polygon, and Binance Smart Chain using environment variables.**
19. **Create a dynamic service class that adjusts Web3 connection parameters based on the selected network.**

### ⚠️ Error Handling & Logging
20. **Create an advanced error handling mechanism in Laravel for Web3 transactions — automatically retry transactions that fail due to gas issues and log all errors.**

---
