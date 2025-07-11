# Implementing Web3 Login (SIWE, Moralis Auth, and Alternatives)

Here's a comprehensive guide to implementing secure Web3 authentication in your dApp:

## 1. Sign-In with Ethereum (SIWE) - Standard Approach

### Frontend Implementation

```typescript
// lib/siwe.ts
import { SiweMessage } from 'siwe'
import { ethers } from 'ethers'

export async function createSiweMessage(
  address: string,
  statement: string,
  chainId: number
) {
  const domain = window.location.host
  const origin = window.location.origin
  const provider = new ethers.providers.Web3Provider(window.ethereum)
  const nonce = await fetchNonce() // Get from your backend

  const message = new SiweMessage({
    domain,
    address,
    statement,
    uri: origin,
    version: '1',
    chainId,
    nonce
  })

  return message.prepareMessage()
}

export async function signInWithEthereum() {
  if (!window.ethereum) throw new Error('No Ethereum provider found')
  
  const provider = new ethers.providers.Web3Provider(window.ethereum)
  const signer = provider.getSigner()
  const address = await signer.getAddress()
  const { chainId } = await provider.getNetwork()

  const message = await createSiweMessage(
    address,
    'Sign in with Ethereum to the app',
    chainId
  )

  const signature = await signer.signMessage(message)
  
  // Verify with backend
  const response = await verifySignature({ message, signature })
  return response.token // JWT or session token
}
```

### Backend Verification (Node.js)

```typescript
// api/auth/verify.ts
import { SiweMessage } from 'siwe'
import jwt from 'jsonwebtoken'

export async function verifySiweSignature(
  message: string,
  signature: string
) {
  const siweMessage = new SiweMessage(message)
  
  try {
    const fields = await siweMessage.validate(signature)
    
    // Additional checks
    if (fields.domain !== process.env.AUTH_DOMAIN) 
      throw new Error('Invalid domain')
    if (fields.nonce !== (await getStoredNonce(fields.address)))
      throw new Error('Invalid nonce')
    if (fields.expirationTime && new Date(fields.expirationTime) <= new Date())
      throw new Error('Expired message')

    // Create JWT
    const token = jwt.sign(
      { address: fields.address, chainId: fields.chainId },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    )

    return { success: true, token }
  } catch (error) {
    return { success: false, error: error.message }
  }
}
```

## 2. Moralis Auth Implementation

### Setup Moralis

```bash
npm install moralis @moralisweb3/auth
```

### Frontend Integration

```typescript
// lib/moralisAuth.ts
import { Moralis } from 'moralis'
import { Auth } from '@moralisweb3/auth'

const moralisAuth = new Auth({
  apiKey: process.env.NEXT_PUBLIC_MORALIS_API_KEY,
  configuration: {
    domain: process.env.NEXT_PUBLIC_AUTH_DOMAIN,
    uri: process.env.NEXT_PUBLIC_AUTH_ORIGIN,
    timeout: 15, // 15 seconds
  }
})

export async function moralisLogin() {
  try {
    const { address, profileId, signature } = await moralisAuth.authenticate()
    
    // Verify with Moralis
    const response = await Moralis.Auth.verify({
      message: moralisAuth.message,
      signature,
      network: 'evm',
    })
    
    // Get or create user in your DB
    const user = await handleUserLogin(address, profileId)
    
    return user
  } catch (error) {
    console.error('Moralis auth failed:', error)
    throw error
  }
}
```

## 3. Alternative: Custom Web3 Auth Flow

### Frontend Signing

```typescript
// lib/web3Auth.ts
export async function web3Login() {
  const provider = new ethers.providers.Web3Provider(window.ethereum)
  const signer = provider.getSigner()
  const address = await signer.getAddress()
  
  // Get nonce from backend
  const { nonce } = await fetch(`/api/auth/nonce?address=${address}`)
    .then(res => res.json())
  
  // Sign message
  const message = `Welcome to MyDApp!\n\nPlease sign to login.\n\nNonce: ${nonce}`
  const signature = await signer.signMessage(message)
  
  // Verify signature
  const { token } = await fetch('/api/auth/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ address, message, signature })
  }).then(res => res.json())
  
  return token
}
```

### Backend Verification

```typescript
// api/auth/verify.ts
import { verifyMessage } from 'ethers/lib/utils'

export async function verifyWeb3Login(
  address: string,
  message: string,
  signature: string
) {
  // Recover address
  const recoveredAddress = verifyMessage(message, signature)
  
  // Verify address matches
  if (recoveredAddress.toLowerCase() !== address.toLowerCase()) {
    throw new Error('Invalid signature')
  }
  
  // Verify nonce is valid
  const nonce = extractNonceFromMessage(message)
  const storedNonce = await getNonceForAddress(address)
  
  if (nonce !== storedNonce) {
    throw new Error('Invalid nonce')
  }
  
  // Create session
  return createSessionForAddress(address)
}
```

## 4. Session Management

### React Context for Auth State

```typescript
// contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react'

type AuthContextType = {
  user: string | null
  login: () => Promise<void>
  logout: () => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  login: async () => {},
  logout: () => {},
  isLoading: false
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check existing session
    const token = localStorage.getItem('web3_token')
    if (token) {
      verifyToken(token).then(address => {
        setUser(address)
        setIsLoading(false)
      })
    } else {
      setIsLoading(false)
    }
  }, [])

  const login = async () => {
    setIsLoading(true)
    try {
      const token = await signInWithEthereum() // Or other method
      localStorage.setItem('web3_token', token)
      const address = await verifyToken(token)
      setUser(address)
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    localStorage.removeItem('web3_token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
```

## 5. Security Best Practices

### Nonce Management
```typescript
// Generate and store nonces
export async function generateNonce(address: string) {
  const nonce = crypto.randomBytes(16).toString('hex')
  await redis.set(`nonce:${address.toLowerCase()}`, nonce, 'EX', 300) // 5min expiry
  return nonce
}
```

### JWT Validation Middleware
```typescript
// middleware/auth.ts
export async function authenticateRequest(req: NextRequest) {
  const token = req.cookies.get('auth_token')?.value
  if (!token) throw new Error('Unauthorized')

  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET!)
    if (typeof payload === 'string' || !payload.address) {
      throw new Error('Invalid token')
    }
    
    return { address: payload.address as string }
  } catch (err) {
    throw new Error('Invalid or expired token')
  }
}
```

## Implementation Checklist

1. [ ] Choose authentication method (SIWE, Moralis, or custom)
2. [ ] Set up frontend signing flow
3. [ ] Implement backend verification
4. [ ] Add nonce generation/validation
5. [ ] Configure session management
6. [ ] Add proper error handling
7. [ ] Implement logout functionality
8. [ ] Add CSRF protection (for SIWE)
9. [ ] Set token expiration policies
10. [ ] Audit security considerations

## Key Considerations

- **Wallet Support**: Works with all Ethereum wallets (MetaMask, WalletConnect, etc.)
- **No Gas Costs**: Users only sign messages, no transactions needed
- **Session Persistence**: Use JWTs or session cookies
- **Cross-Device Support**: Works on mobile via WalletConnect
- **Spam Protection**: Nonce prevents replay attacks
