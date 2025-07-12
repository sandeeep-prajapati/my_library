
---

### **Difficulty Level: Beginner**  
**Objective:** Learn Solidity syntax, basic smart contract structure, and simple functionalities.  

#### **Assignment 1: Hello World Contract**  
- Write a contract that stores and retrieves a string message (`Hello, World!`).  
- Include:  
  - A state variable to store the message.  
  - A function to update the message.  
  - A function to read the message.  

#### **Assignment 2: Simple Token (ERC-20 Basics)**  
- Create a basic token with:  
  - A fixed supply (e.g., 1,000,000 tokens).  
  - A function to transfer tokens between two addresses.  
  - A mapping to track balances.  

#### **Assignment 3: Voting System**  
- Build a contract where:  
  - The owner can add candidates.  
  - Users can vote for a candidate (one vote per address).  
  - A function displays the vote count for each candidate.  

---

### **Difficulty Level: Intermediate**  
**Objective:** Work with security best practices, inheritance, and more complex logic.  

#### **Assignment 4: Multi-Signature Wallet**  
- Create a wallet that requires **M-of-N** approvals for transactions.  
- Features:  
  - Add/remove owners.  
  - Submit, approve, and execute transactions.  
  - Prevent reentrancy attacks.  

#### **Assignment 5: Staking & Rewards**  
- Build a staking contract where:  
  - Users can deposit ETH/tokens.  
  - They earn rewards over time (e.g., 5% APR).  
  - Include withdrawal with penalties for early unstaking.  

#### **Assignment 6: Dutch Auction**  
- Implement a descending-price auction:  
  - The price starts high and decreases over time.  
  - Buyers can purchase at the current price.  
  - The auction ends when all items are sold or time expires.  

---

### **Difficulty Level: Advanced**  
**Objective:** Master gas optimization, complex DeFi logic, and security vulnerabilities.  

#### **Assignment 7: Decentralized Exchange (DEX)**  
- Build a simple AMM (Automated Market Maker) like Uniswap:  
  - Implement `addLiquidity`, `removeLiquidity`, and `swap` functions.  
  - Use the formula `x * y = k` for pricing.  
  - Handle LP (Liquidity Provider) tokens.  

#### **Assignment 8: Flash Loan Arbitrage**  
- Create a contract that:  
  - Takes a flash loan (e.g., from Aave or a mock pool).  
  - Executes an arbitrage between two DEXs.  
  - Repays the loan + fee in one transaction.  

#### **Assignment 9: Upgradeable Smart Contract**  
- Use **Proxy Patterns** (e.g., Transparent Proxy or UUPS):  
  - Deploy a logic contract and a proxy.  
  - Simulate upgrading the contract without losing state.  

---

### **Expert Level (Bonus Challenges)**  
- **Assignment 10:** Optimize a contract for **gas efficiency** (e.g., using assembly/Yul).  
- **Assignment 11:** Implement a **zk-SNARKs-based private transaction** system.  
- **Assignment 12:** Build a **DAO governance system** with voting and delegation.  

---


---

## **Difficulty Level: Beginner**  
**Objective:** Learn basic smart contract interaction with a frontend.  

### **1. Simple Wallet Connection**  
**Task:**  
- Build a React app that connects to MetaMask using Ethers.js.  
- Display the connected wallet address and ETH balance.  

**Skills Learned:**  
- Ethers.js `provider` and `signer`.  
- React state management for wallet connection.  

---

### **2. Token Balance Checker**  
**Task:**  
- Deploy a simple ERC-20 token contract (or use an existing one like DAI).  
- Create a React app where users can:  
  - Enter a token contract address.  
  - Enter their wallet address.  
  - Fetch and display their token balance.  

**Skills Learned:**  
- Interacting with ERC-20 contracts (`balanceOf`).  
- Basic Ethers.js contract calls.  

---

### **3. Basic Transaction Sender**  
**Task:**  
- Build a React app where users can:  
  - Enter an ETH amount and recipient address.  
  - Send ETH transactions via MetaMask.  
  - Display transaction status (pending, success, failure).  

**Skills Learned:**  
- Sending transactions with Ethers.js (`sendTransaction`).  
- Handling transaction receipts.  

---

## **Difficulty Level: Intermediate**  
**Objective:** Work with more complex smart contracts and frontend state management.  

### **4. Todo List dApp (Stateful Smart Contract)**  
**Task:**  
- Create a Solidity contract that stores tasks (`addTask`, `toggleComplete`, `deleteTask`).  
- Build a React frontend to:  
  - Fetch and display tasks for the connected wallet.  
  - Allow adding, toggling, and deleting tasks.  

**Skills Learned:**  
- Reading and writing to a smart contract.  
- React state updates based on blockchain data.  

---

### **5. Staking dApp (with Rewards)**  
**Task:**  
- Write a Solidity staking contract where users can:  
  - Deposit ERC-20 tokens.  
  - Earn rewards over time.  
  - Withdraw staked tokens + rewards.  
- Build a React frontend to interact with it.  

**Skills Learned:**  
- Time-based rewards logic in Solidity.  
- Tracking staking positions in the frontend.  

---

### **6. Multi-Signature Wallet (Advanced Transactions)**  
**Task:**  
- Create a multi-sig wallet contract (requires `N` out of `M` approvals).  
- Build a React dashboard where owners can:  
  - Propose transactions.  
  - Approve/reject transactions.  
  - Execute approved transactions.  

**Skills Learned:**  
- Complex smart contract logic (approvals, execution).  
- Frontend for multi-step transactions.  

---

## **Difficulty Level: Advanced**  
**Objective:** Build full-fledged DeFi or DAO applications.  

### **7. Decentralized Exchange (DEX) like Uniswap**  
**Task:**  
- Implement a Solidity AMM (Automated Market Maker) with:  
  - `addLiquidity`, `removeLiquidity`, `swap` functions.  
  - LP token minting/burning.  
- Build a React UI for swapping tokens and adding liquidity.  

**Skills Learned:**  
- AMM math (`x * y = k`).  
- Advanced Ethers.js interactions.  

---

### **8. DAO Governance dApp (Voting & Proposals)**  
**Task:**  
- Create a Solidity DAO contract with:  
  - Proposal creation.  
  - Voting with governance tokens.  
  - Execution of passed proposals.  
- Build a React frontend to submit/vote on proposals.  

**Skills Learned:**  
- Token-weighted voting.  
- Handling complex governance logic.  

---

### **9. NFT Marketplace (Minting, Buying, Selling)**  
**Task:**  
- Deploy an ERC-721 NFT contract with a marketplace.  
- Features:  
  - Mint NFTs.  
  - List NFTs for sale.  
  - Buy NFTs with ETH.  
- React frontend to browse and trade NFTs.  

**Skills Learned:**  
- NFT standards (ERC-721).  
- Marketplace escrow logic.  

---

## **Bonus: Expert Challenges**  
- **10. Flash Loan Arbitrage Bot** (React dashboard + Solidity executor).  
- **11. Layer-2 dApp** (Using Optimism/Arbitrum + Ethers.js).  
- **12. Gasless Transactions** (Meta-transactions with EIP-2771).  

---

### **Tools to Use:**  
- **Smart Contracts:** Solidity + Hardhat/Foundry.  
- **Frontend:** React + Ethers.js + Vite.  
- **Styling:** TailwindCSS or Chakra UI.  
- **Testing:** Hardhat tests + React testing library.  

---


---

### **🔹 Core & Development Tools**
| Name | Purpose |
|------|---------|
| **[Laravel Sail](https://laravel.com/docs/sail)** | Docker-based local dev environment |
| **[Laravel Herd](https://herd.laravel.com/)** | Fast PHP & Laravel local development (Mac) |
| **[Laravel Valet](https://laravel.com/docs/valet)** | Lightweight dev environment (Mac) |
| **[Laragon](https://laragon.org/)** | Portable dev environment (Windows) |
| **[Laravel Debugbar](https://github.com/barryvdh/laravel-debugbar)** | Debugging toolbar for queries, requests, etc. |
| **[Tinker](https://laravel.com/docs/artisan#tinker)** | REPL for interacting with Laravel via CLI |

---

### **🔹 Authentication & Security**
| Name | Purpose |
|------|---------|
| **[Laravel Breeze](https://laravel.com/docs/starter-kits#laravel-breeze)** | Simple auth scaffolding (Blade/React/Vue) |
| **[Laravel Fortify](https://laravel.com/docs/fortify)** | Backend auth system (headless) |
| **[Laravel Sanctum](https://laravel.com/docs/sanctum)** | API token & SPA authentication |
| **[Laravel Socialite](https://laravel.com/docs/socialite)** | OAuth login (Google, Facebook, etc.) |
| **[Laravel Jetstream](https://jetstream.laravel.com/)** | Advanced auth + Livewire/Inertia stack |
| **[Spatie Laravel-Permission](https://github.com/spatie/laravel-permission)** | Role & Permission management |

---

### **🔹 API Development**
| Name | Purpose |
|------|---------|
| **[Laravel Sanctum](https://laravel.com/docs/sanctum)** | Lightweight API auth |
| **[Laravel Passport](https://laravel.com/docs/passport)** | OAuth2 server implementation |
| **[Laravel API Resources](https://laravel.com/docs/eloquent-resources)** | Transform Eloquent models into JSON |
| **[Dingo API](https://github.com/dingo/api)** (Deprecated but still used) | Advanced API tools |
| **[Laravel JSON:API](https://laravel-json-api.readthedocs.io/)** | JSON:API standard implementation |

---

### **🔹 Database & Eloquent**
| Name | Purpose |
|------|---------|
| **[Laravel Telescope](https://laravel.com/docs/telescope)** | Debugging & monitoring tool |
| **[Laravel Scout](https://laravel.com/docs/scout)** | Full-text search (Algolia, Meilisearch) |
| **[Spatie Laravel-MediaLibrary](https://github.com/spatie/laravel-medialibrary)** | File & media uploads management |
| **[Laravel Excel](https://laravel-excel.com/)** | Import/export Excel & CSV files |
| **[Laravel Backup](https://github.com/spatie/laravel-backup)** | Database & file backups |
| **[Eloquent-Sluggable](https://github.com/cviebrock/eloquent-sluggable)** | Generate URL slugs for models |

---

### **🔹 Frontend & UI**
| Name | Purpose |
|------|---------|
| **[Livewire](https://laravel-livewire.com/)** | Full-stack reactive components |
| **[Inertia.js](https://inertiajs.com/)** | SPA-like apps without API boilerplate |
| **[Laravel Mix](https://laravel-mix.com/)** | Webpack wrapper for asset compilation |
| **[Laravel Vite](https://laravel.com/docs/vite)** | Modern frontend build tool (replacing Mix) |
| **[Tailwind CSS](https://tailwindcss.com/)** | Utility-first CSS framework |
| **[Alpine.js](https://alpinejs.dev/)** | Lightweight JavaScript reactivity |

---

### **🔹 Testing & Debugging**
| Name | Purpose |
|------|---------|
| **[Pest](https://pestphp.com/)** | Modern PHP testing framework |
| **[Laravel Dusk](https://laravel.com/docs/dusk)** | Browser automation testing |
| **[PHPUnit](https://phpunit.de/)** | Default Laravel testing framework |
| **[Laravel Telescope](https://laravel.com/docs/telescope)** | Debugging & request monitoring |
| **[Clockwork](https://github.com/itsgoingd/clockwork)** | Alternative to Debugbar |

---

### **🔹 Deployment & DevOps**
| Name | Purpose |
|------|---------|
| **[Laravel Forge](https://forge.laravel.com/)** | Server provisioning & deployment |
| **[Laravel Envoyer](https://envoyer.io/)** | Zero-downtime PHP deployment |
| **[Laravel Horizon](https://laravel.com/docs/horizon)** | Redis queue dashboard |
| **[Laravel Octane](https://laravel.com/docs/octane)** | High-performance app server (Swoole/RoadRunner) |
| **[Deployer](https://deployer.org/)** | PHP deployment tool |

---

### **🔹 Payments & E-Commerce**
| Name | Purpose |
|------|---------|
| **[Laravel Cashier](https://laravel.com/docs/cashier)** | Stripe & Paddle subscriptions |
| **[Laravel Spark](https://spark.laravel.com/)** | SaaS boilerplate |
| **[Laravel Pay](https://github.com/laravel/pay)** | Multi-gateway payments (Stripe, PayPal, etc.) |
| **[Bagisto](https://bagisto.com/en/)** | Laravel e-commerce platform |

---

### **🔹 Useful Utilities**
| Name | Purpose |
|------|---------|
| **[Laravel Helpers](https://laravel.com/docs/helpers)** | Built-in helper functions |
| **[Laravel Collections](https://laravel.com/docs/collections)** | Powerful array/object manipulation |
| **[Laravel Task Scheduling](https://laravel.com/docs/scheduling)** | Cron-like job scheduler |
| **[Laravel Notifications](https://laravel.com/docs/notifications)** | Email/SMS/Slack notifications |
| **[Laravel Mail](https://laravel.com/docs/mail)** | Email sending (SMTP, Mailgun, etc.) |

---
