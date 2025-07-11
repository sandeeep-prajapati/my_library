# Preventing Phishing & Wallet-Draining Attacks in dApps

Protecting users from malicious attacks is critical for any Web3 application. Here's a comprehensive strategy to safeguard your dApp against phishing and wallet-draining scams:

## 1. Secure Transaction Best Practices

### Transaction Simulation & Warnings
```javascript
// Implement transaction preview before signing
function displayTransactionPreview(tx) {
  return {
    recipient: tx.to,
    value: ethers.utils.formatEther(tx.value),
    gas: tx.gasLimit.toString(),
    data: tx.data ? `0x${tx.data.slice(0, 12)}...` : 'None',
    chainId: tx.chainId
  };
}

// Show confirmation dialog with decoded data
async function confirmTransaction(txRequest) {
  const preview = displayTransactionPreview(txRequest);
  return await showModal({
    title: "Confirm Transaction",
    content: TransactionPreview(preview),
    danger: isUnusualTransaction(txRequest)
  });
}
```

### Address Poisoning Protection
```javascript
// Verify recipient addresses against known contracts
const KNOWN_MALICIOUS_ADDRESSES = [
  '0x123...drainer',
  '0x456...phish'
];

function validateRecipient(address) {
  const checks = [
    isZeroAddress(address),
    isContract(address),
    KNOWN_MALICIOUS_ADDRESSES.includes(address.toLowerCase())
  ];
  
  if (checks.some(Boolean)) {
    throw new Error('Suspicious recipient address');
  }
}
```

## 2. Frontend Security Measures

### DOM Integrity Checks
```javascript
// Detect DOM tampering attempts
setInterval(() => {
  const buttons = document.querySelectorAll('[data-testid="connect-button"]');
  if (buttons.length > 1) {
    alert('Security warning: Duplicate connect buttons detected!');
    window.location.reload();
  }
}, 5000);
```

### Secure Wallet Connection Flow
```typescript
// Implement strict connection validation
async function connectWallet() {
  // Verify we're connecting to the real provider
  if (!window.ethereum?.isMetaMask && !window.ethereum?.isTrust) {
    throw new Error('Unrecognized wallet provider');
  }

  // Request minimal permissions
  const accounts = await window.ethereum.request({
    method: 'eth_requestAccounts',
    params: [{
      eth_accounts: {},
      // Don't request unnecessary permissions
    }]
  });

  // Verify chain
  const chainId = await window.ethereum.request({ method: 'eth_chainId' });
  if (!SUPPORTED_CHAINS.includes(chainId)) {
    await switchToDefaultChain();
  }
}
```

## 3. Backend Protections

### Transaction Validation API
```javascript
// Server-side transaction validation endpoint
app.post('/api/validate-tx', async (req, res) => {
  const { txData, userAddress } = req.body;
  
  const analysis = await securityService.analyzeTransaction(
    txData,
    userAddress
  );

  if (analysis.riskScore > 0.7) {
    return res.status(400).json({
      warning: 'High risk transaction detected',
      details: analysis.threats
    });
  }

  res.json({ approved: true });
});
```

### Threat Intelligence Feeds
```javascript
// Subscribe to real-time threat feeds
const threatFeed = new WebSocket('wss://api.blocksec.com/threat-feed');

threatFeed.onmessage = (event) => {
  const { maliciousContracts, phishingSites } = JSON.parse(event.data);
  updateBlacklist(maliciousContracts);
  warnUsersAboutSites(phishingSites);
};
```

## 4. User Education Features

### In-App Security Warnings
```javascript
// Show warnings for suspicious activities
function SecurityAlert({ type }) {
  const alerts = {
    highValue: {
      icon: '⚠️',
      message: 'You're about to send a large amount of tokens'
    },
    newContract: {
      icon: '🔍',
      message: 'This contract was deployed recently - verify before interacting'
    },
    approval: {
      icon: '🛑',
      message: 'You're granting unlimited token access - revoke after use'
    }
  };

  return (
    <div className={`security-alert ${type}`}>
      {alerts[type].icon} {alerts[type].message}
    </div>
  );
}
```

### Transaction Simulation
```javascript
// Show expected outcome before signing
async function simulateTransfer(token, amount, recipient) {
  const result = await token.callStatic.transfer(recipient, amount);
  
  return {
    balanceChange: await token.balanceOf(userAddress) - amount,
    recipientBalanceChange: (await token.balanceOf(recipient)) + amount,
    success: result
  };
}
```

## 5. Smart Contract Safeguards

### Approval Patterns
```solidity
// Use permit instead of approve when possible
function safeApprove(
  IERC20 token,
  address spender,
  uint256 amount,
  uint256 deadline,
  uint8 v,
  bytes32 r,
  bytes32 s
) external {
  token.permit(msg.sender, address(this), amount, deadline, v, r, s);
  token.transferFrom(msg.sender, spender, amount);
}
```

### Time-Limited Approvals
```solidity
// Expiring approvals
mapping(address => mapping(address => uint256)) private _approvals;
mapping(address => mapping(address => uint256)) private _approvalExpiry;

function approveWithExpiry(address spender, uint256 amount, uint256 expiryTime) external {
  _approvals[msg.sender][spender] = amount;
  _approvalExpiry[msg.sender][spender] = expiryTime;
}

function allowance(address owner, address spender) public view returns (uint256) {
  if (block.timestamp > _approvalExpiry[owner][spender]) {
    return 0;
  }
  return _approvals[owner][spender];
}
```

## 6. Monitoring & Analytics

### Suspicious Activity Detection
```javascript
// Track unusual behavior patterns
class BehaviorMonitor {
  constructor() {
    this.actions = [];
  }

  logAction(type, metadata) {
    this.actions.push({ type, timestamp: Date.now(), metadata });
    this.checkPatterns();
  }

  checkPatterns() {
    // Detect rapid-fire approvals
    const recentApprovals = this.actions.filter(a => 
      a.type === 'approval' && 
      Date.now() - a.timestamp < 5000
    );
    
    if (recentApprovals.length > 3) {
      triggerSecurityLock();
    }
  }
}
```

## Implementation Checklist

1. [ ] Integrate transaction previews before signing
2. [ ] Implement address poisoning protection
3. [ ] Add DOM integrity monitoring
4. [ ] Create server-side transaction validation
5. [ ] Subscribe to threat intelligence feeds
6. [ ] Develop in-app security education
7. [ ] Use time-limited approvals in contracts
8. [ ] Set up behavior monitoring
9. [ ] Regularly update malicious address databases
10. [ ] Conduct periodic security audits
