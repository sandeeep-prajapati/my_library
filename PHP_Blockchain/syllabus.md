
---

### **1. Laravel + Web3 Basics**  
1. **01_laravel_web3_connection.md**  
   - How to connect Laravel to Ethereum/BSC using **Alchemy/Infura**?  
   - Best practices for storing `.env` variables (RPC URL, private keys).  

2. **02_web3_laravel_package.md**  
   - Compare `web3.php` vs `EthereumPHP`—which is better for Laravel?  
   - How to create a **custom Laravel Service Provider** for Web3 connections?  

3. **03_eth_block_number.md**  
   - Build a Laravel API endpoint that fetches the latest **Ethereum block number**.  
   - Handle errors (timeout, rate limits) gracefully.  

4. **04_eth_balance_checker.md**  
   - Create a Laravel command to check **ETH/BNB balance** of any address.  

5. **05_gas_price_tracker.md**  
   - Fetch real-time **gas prices** (low, medium, high) and display in Laravel.  

---

### **2. Smart Contract Integration**  
6. **06_deploy_smart_contract.md**  
   - How to **deploy a Solidity contract** from Laravel using `web3.php`?  

7. **07_call_contract_function.md**  
   - How to call a **read-only** Solidity function (e.g., `balanceOf`) from Laravel?  

8. **08_send_transaction.md**  
   - How to send a **transaction** (e.g., transfer ERC-20 tokens) via Laravel?  

9. **09_events_listener.md**  
   - How to listen for **Smart Contract Events** (e.g., `Transfer`) in Laravel?  

10. **10_verify_smart_contract.md**  
   - Automate **Etherscan verification** of contracts after deployment via Laravel.  

---

### **3. Wallet & Auth (Web3 Login)**  
11. **11_metamask_login.md**  
   - How to implement **"Login with MetaMask"** in Laravel?  

12. **12_jwt_web3_auth.md**  
   - How to issue **JWT tokens** after Web3 authentication?  

13. **13_nonce_based_auth.md**  
   - Secure Web3 login using **signature nonce verification** in Laravel.  

14. **14_multisig_wallet.md**  
   - How to interact with a **Gnosis Safe MultiSig Wallet** from Laravel?  

15. **15_php_hd_wallets.md**  
   - Generate **HD wallets** (BIP39/BIP44) in Laravel for user accounts.  

---

### **4. DeFi & Tokens**  
16. **16_erc20_balance.md**  
   - Fetch **ERC-20 token balances** for a wallet in Laravel.  

17. **17_uniswap_price.md**  
   - How to fetch **token prices** from Uniswap/PancakeSwap in Laravel?  

18. **18_swap_tokens.md**  
   - Build a **token swap** feature (1inch/0x API) in Laravel.  

19. **19_staking_dapp.md**  
   - How to interact with a **staking smart contract** from Laravel?  

20. **20_liquidity_pools.md**  
   - Fetch **LP token balances** (Uniswap/PancakeSwap) via Laravel.  

---

### **5. NFTs in Laravel**  
21. **21_nft_minting.md**  
   - How to **mint an NFT** from Laravel (ERC-721/ERC-1155)?  

22. **22_nft_marketplace.md**  
   - Build an **NFT marketplace backend** (listings, bids) in Laravel.  

23. **23_opensea_api.md**  
   - How to fetch **NFT metadata** from OpenSea in Laravel?  

24. **24_nft_royalties.md**  
   - Handle **royalty payouts** for NFT sales in Laravel.  

25. **25_soulbound_tokens.md**  
   - Implement **Soulbound Tokens (SBTs)** with Laravel.  

---

### **6. Security & Optimization**  
26. **26_private_key_security.md**  
   - Best ways to **securely store private keys** in Laravel.  

27. **27_reentrancy_guard.md**  
   - How to prevent **reentrancy attacks** in Solidity + Laravel apps?  

28. **28_gas_optimization.md**  
   - Optimize **gas costs** for transactions sent from Laravel.  

29. **29_rate_limiting.md**  
   - Implement **rate limiting** for blockchain RPC calls in Laravel.  

30. **30_mainnet_deployment.md**  
   - How to **deploy a Laravel DApp** on Ethereum/BSC mainnet?  

---