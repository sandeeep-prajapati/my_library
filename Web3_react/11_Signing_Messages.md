# Implementing Message Signing with MetaMask for Authentication

Message signing is a secure way to authenticate users without requiring gas fees or on-chain transactions. Here's how to implement it with MetaMask:

## Basic Implementation Steps

### 1. Connect to MetaMask First
```javascript
async function connectMetaMask() {
  if (window.ethereum) {
    try {
      const accounts = await window.ethereum.request({ 
        method: 'eth_requestAccounts' 
      });
      return accounts[0]; // Return the connected address
    } catch (error) {
      console.error("User denied account access", error);
    }
  } else {
    alert("Please install MetaMask!");
  }
}
```

### 2. Sign a Message
```javascript
async function signMessage(message, address) {
  try {
    const signature = await window.ethereum.request({
      method: 'personal_sign',
      params: [message, address],
    });
    return signature;
  } catch (err) {
    console.error("Signing failed:", err);
    return null;
  }
}
```

## Complete Authentication Flow

### Frontend Implementation
```javascript
async function authenticateWithMetaMask() {
  // 1. Connect wallet
  const userAddress = await connectMetaMask();
  if (!userAddress) return;
  
  // 2. Generate a unique nonce from your backend
  const nonceResponse = await fetch('/api/auth/nonce', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ address: userAddress })
  });
  const { nonce } = await nonceResponse.json();
  
  // 3. Create and sign message
  const message = `Please sign this message to authenticate. Nonce: ${nonce}`;
  const signature = await signMessage(message, userAddress);
  
  // 4. Verify with backend
  const authResponse = await fetch('/api/auth/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ address: userAddress, message, signature })
  });
  
  if (authResponse.ok) {
    // Authentication successful - receive JWT or session token
    const { token } = await authResponse.json();
    localStorage.setItem('authToken', token);
    console.log("Authentication successful!");
  }
}
```

### Backend Verification (Node.js Example)
```javascript
const ethers = require('ethers');

async function verifySignature(address, message, signature) {
  try {
    // Recover the address from the signature
    const recoveredAddress = ethers.utils.verifyMessage(message, signature);
    
    // Compare with claimed address (case-insensitive)
    return recoveredAddress.toLowerCase() === address.toLowerCase();
  } catch (err) {
    console.error("Verification failed:", err);
    return false;
  }
}

// Express route handler
app.post('/api/auth/verify', async (req, res) => {
  const { address, message, signature } = req.body;
  
  const isValid = await verifySignature(address, message, signature);
  
  if (isValid) {
    // Create session/JWT token
    const token = createAuthToken(address);
    res.json({ success: true, token });
  } else {
    res.status(401).json({ error: "Invalid signature" });
  }
});
```

## Security Considerations

1. **Nonce Protection**:
   - Generate a unique nonce for each authentication attempt
   - Expire nonces after short time periods (5-15 minutes)
   - Store nonces server-side

2. **Message Format**:
   - Include your dApp name in the message
   - Clearly state this is for authentication
   - Add nonce to prevent replay attacks

3. **Additional Checks**:
   - Verify the message was signed recently
   - Consider adding chain ID to prevent cross-chain replay

## UX Improvements

1. **Customizing the Sign Request**:
```javascript
const message = `Welcome to MyDApp!

Click to sign in and accept the Terms of Service.

This request will not trigger a blockchain transaction or cost any gas fees.

Wallet address:
${userAddress}

Nonce:
${nonce}`;
```

2. **Error Handling**:
   - Handle user rejection gracefully
   - Detect if MetaMask is locked
   - Provide clear instructions

## Alternative: Sign-In with Ethereum (SIWE)

For standardized authentication, consider using [EIP-4361](https://login.xyz):
```javascript
import { SiweMessage } from 'siwe';

// Generate SIWE message
const siweMessage = new SiweMessage({
  domain: window.location.host,
  address: userAddress,
  statement: 'Sign in with Ethereum to the app.',
  uri: window.location.origin,
  version: '1',
  chainId: 1, // Mainnet
  nonce: await generateNonce(),
});
const message = siweMessage.prepareMessage();

// Then proceed with signing as before
```
---
