### 🔹 What is **ERC-1155**?

**ERC-1155** is a **multi-token standard** that allows a **single smart contract** to manage:
- **Fungible tokens** (like coins),
- **Non-fungible tokens (NFTs)** (like unique items),
- **Semi-fungible tokens** (like game skins that are rare but not unique).

➡️ It was proposed by **Enjin** in 2018 to address limitations of ERC-721 and ERC-20.

---

### 🔄 Key Differences: **ERC-1155 vs. ERC-721**

| Feature | ERC-721 | ERC-1155 |
|--------|---------|----------|
| Token Type | Only Non-Fungible | Both Fungible & Non-Fungible |
| Contract Efficiency | 1 NFT = 1 contract | Many tokens in 1 contract |
| Transfer Function | Transfers **1 token at a time** | Transfers **multiple tokens in a batch** |
| Gas Fees | Higher (more transactions) | Lower (batching = cheaper) |
| Metadata | One URI per token | Shared base URI with dynamic `{id}` |
| Use Case | Art, collectibles | Games, marketplaces, mixed assets |

---

### 🔹 How ERC-1155 Works (Conceptually)

- Tokens are identified by a **`uint256` ID**.
- The same ID can have **multiple copies** (for fungible/semi-fungible items).
- NFTs are created by ensuring **supply = 1** for a given ID.

---

### 🔧 Core Functions of ERC-1155

```solidity
function balanceOf(address account, uint256 id) external view returns (uint256);
function safeTransferFrom(address from, address to, uint256 id, uint256 amount, bytes data) external;
function safeBatchTransferFrom(address from, address to, uint256[] ids, uint256[] amounts, bytes data) external;
```

- **`balanceOf()`**: Returns how many copies of a token an address holds.
- **Batch Transfers**: Move multiple token types in a single transaction = **super gas-efficient**.

---

### 🔮 When to Use ERC-1155 Instead of ERC-721

| Scenario | Best Standard |
|---------|----------------|
| You’re creating unique digital art pieces | **ERC-721** |
| You’re building a game with **weapons, skins, and collectibles** | ✅ **ERC-1155** |
| You want to **mint or transfer tokens in bulk** (e.g., in-game currencies + items) | ✅ **ERC-1155** |
| You're making a **marketplace** that supports many token types | ✅ **ERC-1155** |
| You want simplicity, only dealing with **one-off NFTs** | **ERC-721** |

---

### 🎮 Real Example: A Game Inventory

With **ERC-1155**, you can have:
- Token ID 1 → 🔥 1000 swords (fungible)
- Token ID 2 → 🎨 1 unique crown (non-fungible)
- Token ID 3 → 🧪 500 health potions (semi-fungible)

All in **one contract**.

---

### 📦 Metadata Handling

ERC-1155 uses a **URI with `{id}` substitution**:
```json
https://gameitems.io/api/item/{id}.json
```
The client replaces `{id}` with the token ID (in hex).

---

### ⚡ Summary

| Feature | ERC-721 | ERC-1155 |
|--------|---------|----------|
| Use Case | Unique NFTs | Mixed assets, batch minting |
| Token Type | One per contract | Multiple |
| Gas Efficient? | ❌ No | ✅ Yes |
| Batch Transfer? | ❌ No | ✅ Yes |
| Metadata URI | Per token | Shared with dynamic ID |
