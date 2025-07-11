# Implementing WalletConnect for Mobile Wallet Support

Adding WalletConnect to your dApp enables seamless connections with 200+ mobile wallets. Here's a comprehensive implementation guide:

## 1. Setup WalletConnect Project

First, create a project on [WalletConnect Cloud](https://cloud.walletconnect.com/) to get your Project ID.

```bash
npm install @walletconnect/ethereum-provider @web3modal/ethereum @web3modal/react
```

## 2. Configure Web3Modal (v2 Recommended)

```typescript
// lib/walletConnect.ts
import { createWeb3Modal, defaultWagmiConfig } from '@web3modal/wagmi/react'
import { WalletConnectConnector } from 'wagmi/connectors/walletConnect'
import { mainnet, polygon, arbitrum } from 'wagmi/chains'

// 1. Define chains
const chains = [mainnet, polygon, arbitrum] as const
const projectId = 'YOUR_WALLETCONNECT_PROJECT_ID'

// 2. Create wagmiConfig
const metadata = {
  name: 'My dApp',
  description: 'My dApp description',
  url: 'https://mydapp.com',
  icons: ['https://mydapp.com/logo.png']
}

const wagmiConfig = defaultWagmiConfig({ chains, projectId, metadata })

// 3. Create modal
createWeb3Modal({
  wagmiConfig,
  projectId,
  chains,
  themeVariables: {
    '--w3m-accent': '#3b82f6',
    '--w3m-font-size-master': '16px'
  }
})
```

## 3. Integrate with Your App

```tsx
// App.tsx
import { WagmiConfig } from 'wagmi'
import { Web3Modal } from '@web3modal/react'
import { walletConnectConfig } from './lib/walletConnect'

function App() {
  return (
    <>
      <WagmiConfig config={walletConnectConfig.wagmiConfig}>
        {/* Your app components */}
        <ConnectButton />
      </WagmiConfig>

      <Web3Modal 
        config={walletConnectConfig.web3ModalConfig}
      />
    </>
  )
}
```

## 4. Create a Custom Connect Button

```tsx
// components/ConnectButton.tsx
import { useWeb3Modal } from '@web3modal/react'
import { useAccount, useDisconnect } from 'wagmi'

export function ConnectButton() {
  const { open } = useWeb3Modal()
  const { address, isConnected } = useAccount()
  const { disconnect } = useDisconnect()

  return (
    <button
      onClick={() => isConnected ? disconnect() : open()}
      className="connect-button"
    >
      {isConnected ? (
        `Connected: ${address?.slice(0, 6)}...${address?.slice(-4)}`
      ) : (
        'Connect Wallet'
      )}
    </button>
  )
}
```

## 5. Handle WalletConnect Session Management

```typescript
// hooks/useWalletConnect.ts
import { useWeb3Modal } from '@web3modal/react'
import { useAccount, useConnect, useDisconnect } from 'wagmi'

export function useWalletConnect() {
  const { open } = useWeb3Modal()
  const { address, connector, isConnected } = useAccount()
  const { disconnect } = useDisconnect()
  const { connect, connectors, error, isLoading, pendingConnector } = useConnect()

  const connectWallet = async () => {
    try {
      await open()
    } catch (err) {
      console.error('WalletConnect error:', err)
    }
  }

  return {
    address,
    isConnected,
    connect: connectWallet,
    disconnect,
    activeConnector: connector,
    error,
    isLoading
  }
}
```

## 6. Add Mobile-Specific Optimizations

### Deep Linking for Mobile Apps

```typescript
// Add to your walletConnect configuration
const walletConnectConnector = new WalletConnectConnector({
  options: {
    projectId,
    showQrModal: false, // We're using Web3Modal's QR
    qrcode: true,
    metadata,
    qrModalOptions: {
      themeVariables: {
        '--wcm-z-index': '9999'
      },
      mobileWallets: [
        {
          id: 'metamask',
          name: 'MetaMask',
          links: {
            native: 'metamask://',
            universal: 'https://metamask.app.link'
          }
        },
        {
          id: 'trust',
          name: 'Trust Wallet',
          links: {
            native: 'trust://',
            universal: 'https://link.trustwallet.com'
          }
        }
      ]
    }
  }
})
```

### Session Persistence

```typescript
// lib/session.ts
export const persistSession = (session: any) => {
  localStorage.setItem('walletconnect', JSON.stringify(session))
}

export const getPersistedSession = () => {
  const session = localStorage.getItem('walletconnect')
  return session ? JSON.parse(session) : null
}

export const clearSession = () => {
  localStorage.removeItem('walletconnect')
}
```

## 7. Full Implementation Example

```tsx
// providers/WalletConnectProvider.tsx
'use client'

import { ReactNode, useEffect } from 'react'
import { createWeb3Modal, defaultWagmiConfig } from '@web3modal/wagmi/react'
import { WagmiConfig } from 'wagmi'
import { arbitrum, mainnet, polygon } from 'wagmi/chains'
import { Web3Modal } from '@web3modal/react'

const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID || ''

const metadata = {
  name: 'My dApp',
  description: 'My dApp description',
  url: 'https://mydapp.com',
  icons: ['https://mydapp.com/logo.png']
}

const chains = [mainnet, polygon, arbitrum]
const wagmiConfig = defaultWagmiConfig({ chains, projectId, metadata })

createWeb3Modal({
  wagmiConfig,
  projectId,
  chains,
  themeMode: 'dark',
  featuredWalletIds: [
    'c57ca95b47569778a828d19178114f4db188b89b763c899ba0be274e97267d96', // MetaMask
    '4622a2b2d6af1c9844944291e5e7351a6aa24cd7b23099efac1b2fd875da31a0' // Trust Wallet
  ]
})

export function WalletConnectProvider({ children }: { children: ReactNode }) {
  return (
    <WagmiConfig config={wagmiConfig}>
      {children}
      <Web3Modal />
    </WagmiConfig>
  )
}
```

## 8. Error Handling and UX Improvements

```tsx
// components/WalletConnectFeedback.tsx
import { useWeb3ModalState } from '@web3modal/react'

export function WalletConnectFeedback() {
  const { open, selectedNetworkId } = useWeb3ModalState()

  useEffect(() => {
    if (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to connect wallet'
      )
    }
  }, [error])

  return (
    <div className="wallet-connect-feedback">
      {isLoading && (
        <div className="connection-loading">
          <Spinner />
          <p>Connecting to {pendingConnector?.name}...</p>
        </div>
      )}
      
      {selectedNetworkId && (
        <NetworkIndicator chainId={selectedNetworkId} />
      )}
    </div>
  )
}
```

## Implementation Checklist

1. [ ] Register project on WalletConnect Cloud
2. [ ] Install required packages
3. [ ] Configure Web3Modal with your chains
4. [ ] Wrap app with WagmiConfig and Web3Modal
5. [ ] Implement connect/disconnect logic
6. [ ] Add mobile wallet deep links
7. [ ] Set up session persistence
8. [ ] Add error handling and loading states
9. [ ] Test with multiple mobile wallets
10. [ ] Optimize for both desktop and mobile

## Troubleshooting Tips

1. **QR Code Not Appearing**: Ensure you're not in a strict Content-Security-Policy environment
2. **Connection Drops**: Implement session persistence and reconnection logic
3. **Chain Mismatches**: Always verify the connected chain matches your dApp's requirements
4. **Mobile Linking Issues**: Test with wallet apps installed and via browser

This implementation provides a robust WalletConnect integration that works across all devices and maintains a great user experience. Would you like me to elaborate on any specific aspect of the integration?