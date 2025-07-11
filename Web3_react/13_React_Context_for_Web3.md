# Managing Global Web3 State with React Context

Using React Context for Web3 state management provides a clean way to share wallet connections, network info, and other blockchain data across your entire dApp. Here's a comprehensive implementation:

## 1. Create the Web3 Context

```typescript
// contexts/Web3Context.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { ethers } from 'ethers';

type Web3State = {
  provider?: ethers.providers.Web3Provider;
  signer?: ethers.Signer;
  account: string;
  chainId: number;
  connectWallet: () => Promise<void>;
  disconnectWallet: () => void;
  isConnected: boolean;
  error?: string;
};

const Web3Context = createContext<Web3State>({
  account: '',
  chainId: 0,
  connectWallet: async () => {},
  disconnectWallet: () => {},
  isConnected: false,
});

export const useWeb3 = () => useContext(Web3Context);
```

## 2. Implement the Provider Component

```typescript
// contexts/Web3Context.tsx (continued)
export const Web3Provider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<Omit<Web3State, 'connectWallet' | 'disconnectWallet'>>({
    account: '',
    chainId: 0,
    isConnected: false,
  });

  const connectWallet = async () => {
    try {
      if (!window.ethereum) throw new Error('No Ethereum provider detected');
      
      const provider = new ethers.providers.Web3Provider(window.ethereum);
      const accounts = await provider.send('eth_requestAccounts', []);
      const signer = provider.getSigner();
      const network = await provider.getNetwork();

      setState({
        provider,
        signer,
        account: accounts[0],
        chainId: network.chainId,
        isConnected: true,
        error: undefined,
      });

      // Set up event listeners
      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);
      
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to connect wallet',
      }));
    }
  };

  const disconnectWallet = () => {
    if (window.ethereum?.removeListener) {
      window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
      window.ethereum.removeListener('chainChanged', handleChainChanged);
    }
    
    setState({
      provider: undefined,
      signer: undefined,
      account: '',
      chainId: 0,
      isConnected: false,
    });
  };

  const handleAccountsChanged = (accounts: string[]) => {
    if (accounts.length === 0) {
      disconnectWallet();
    } else {
      setState(prev => ({
        ...prev,
        account: accounts[0],
      }));
    }
  };

  const handleChainChanged = (chainId: string) => {
    setState(prev => ({
      ...prev,
      chainId: parseInt(chainId, 16),
    }));
  };

  // Check if wallet is already connected on mount
  useEffect(() => {
    const checkConnectedWallet = async () => {
      if (window.ethereum) {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        const accounts = await provider.listAccounts();
        if (accounts.length > 0) {
          const network = await provider.getNetwork();
          setState({
            provider,
            signer: provider.getSigner(),
            account: accounts[0],
            chainId: network.chainId,
            isConnected: true,
          });
        }
      }
    };

    checkConnectedWallet();
  }, []);

  return (
    <Web3Context.Provider value={{
      ...state,
      connectWallet,
      disconnectWallet,
    }}>
      {children}
    </Web3Context.Provider>
  );
};
```

## 3. Wrap Your Application with the Provider

```typescript
// App.tsx
import { Web3Provider } from './contexts/Web3Context';

function App() {
  return (
    <Web3Provider>
      <YourAppContent />
    </Web3Provider>
  );
}
```

## 4. Usage in Components

### Basic Connection Example
```typescript
// components/ConnectButton.tsx
import { useWeb3 } from '../contexts/Web3Context';

export const ConnectButton = () => {
  const { account, isConnected, connectWallet, disconnectWallet } = useWeb3();

  return (
    <button 
      onClick={isConnected ? disconnectWallet : connectWallet}
      className="wallet-button"
    >
      {isConnected ? 
        `Connected: ${account.substring(0, 6)}...${account.substring(38)}` : 
        'Connect Wallet'}
    </button>
  );
};
```

### Chain-Aware Component
```typescript
// components/ChainInfo.tsx
import { useWeb3 } from '../contexts/Web3Context';

export const ChainInfo = () => {
  const { chainId, isConnected } = useWeb3();
  
  const getChainName = (id: number) => {
    switch(id) {
      case 1: return 'Ethereum Mainnet';
      case 5: return 'Goerli Testnet';
      case 137: return 'Polygon Mainnet';
      // Add other chains as needed
      default: return `Unknown Chain (ID: ${chainId})`;
    }
  };

  return isConnected ? (
    <div className="chain-info">
      Network: {getChainName(chainId)}
    </div>
  ) : null;
};
```

## Advanced Enhancements

### 1. Add Multi-Chain Support
```typescript
// Update the connectWallet function
const switchChain = async (chainId: string) => {
  try {
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId }],
    });
  } catch (switchError) {
    // If chain isn't added, prompt user to add it
    if (switchError.code === 4902) {
      await window.ethereum.request({
        method: 'wallet_addEthereumChain',
        params: [CHAIN_CONFIGS[chainId]],
      });
    }
  }
};
```

### 2. Persist State with LocalStorage
```typescript
// Add to Web3Provider useEffect
useEffect(() => {
  if (state.isConnected) {
    localStorage.setItem('web3Connection', 'connected');
  } else {
    localStorage.removeItem('web3Connection');
  }
}, [state.isConnected]);
```

### 3. Add Custom Hooks for Common Patterns
```typescript
// hooks/useContract.ts
import { useWeb3 } from '../contexts/Web3Context';
import { Contract } from 'ethers';

export const useContract = (address: string, abi: any) => {
  const { provider, signer } = useWeb3();
  
  if (!provider) throw new Error('Provider not initialized');
  
  return new Contract(
    address,
    abi,
    signer || provider
  );
};
```

## Error Handling Best Practices

1. **Add error boundaries** for components using Web3
2. **Implement retry logic** for failed transactions
3. **Provide user-friendly messages** for common errors like:
   - Wrong network
   - Rejected transactions
   - Insufficient funds

```typescript
// Enhanced error handling in connectWallet
const connectWallet = async () => {
  try {
    // ... existing code ...
  } catch (error) {
    let errorMessage = 'Failed to connect wallet';
    
    if (error.code === 4001) {
      errorMessage = 'Connection rejected by user';
    } else if (error.code === -32002) {
      errorMessage = 'Already processing wallet request';
    }
    
    setState(prev => ({ ...prev, error: errorMessage }));
  }
};
```

This implementation provides a robust foundation for Web3 state management that:
- Handles wallet connection/disconnection
- Tracks account and chain changes
- Provides clean access via hooks
- Supports TypeScript for type safety
- Can be easily extended for additional functionality
---
