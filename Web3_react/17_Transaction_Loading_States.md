# Designing Smooth UX for Pending Blockchain Transactions

Creating a polished user experience for pending transactions is crucial in Web3 applications. Here's a comprehensive approach to handle loading states, notifications, and transaction lifecycle management:

## 1. Transaction Status Tracking System

### Centralized Transaction Store
```typescript
// contexts/TransactionContext.tsx
type Transaction = {
  hash: string;
  status: 'pending' | 'confirmed' | 'failed';
  message: string;
  timestamp: number;
  chainId: number;
  action?: string; // e.g., 'mint', 'swap'
};

type TransactionContextType = {
  transactions: Transaction[];
  addTransaction: (tx: Omit<Transaction, 'status' | 'timestamp'>) => void;
  updateTransaction: (hash: string, updates: Partial<Transaction>) => void;
};

const TransactionContext = createContext<TransactionContextType>(null!);
```

## 2. Visual Feedback Components

### Animated Transaction Toast
```tsx
// components/TransactionToast.tsx
const statusIcons = {
  pending: <Spinner size={16} />,
  confirmed: <CheckCircle size={16} />,
  failed: <XCircle size={16} />,
};

const TransactionToast = ({ tx }: { tx: Transaction }) => (
  <div className={`toast ${tx.status}`}>
    <div className="toast-header">
      {statusIcons[tx.status]}
      <span>{tx.action || 'Transaction'} {tx.status}</span>
      <ExternalLink href={`${getBlockExplorer(tx.chainId)}/tx/${tx.hash}`} />
    </div>
    <div className="toast-body">
      <p>{tx.message}</p>
      {tx.status === 'pending' && (
        <ProgressBar estimatedTime={getEstimatedTime(tx.chainId)} />
      )}
    </div>
  </div>
);
```

### Interactive Transaction History Panel
```tsx
// components/TransactionHistory.tsx
const TransactionHistory = () => {
  const { transactions } = useTransactions();
  
  return (
    <div className="transaction-history">
      <h3>Recent Activity</h3>
      {transactions.slice(0, 5).map(tx => (
        <TransactionItem key={tx.hash} tx={tx} />
      ))}
    </div>
  );
};
```

## 3. Chain-Aware Progress Indicators

### Dynamic Gas Estimation Feedback
```tsx
// components/GasEstimationFeedback.tsx
const GasEstimationFeedback = ({ txRequest }) => {
  const [estimation, setEstimation] = useState<{
    gasLimit: BigNumber;
    gasPrice: BigNumber;
    timeEstimate: number;
  }>();

  useEffect(() => {
    const estimate = async () => {
      const [gasLimit, gasPrice] = await Promise.all([
        provider.estimateGas(txRequest),
        provider.getGasPrice(),
      ]);
      
      setEstimation({
        gasLimit,
        gasPrice,
        timeEstimate: calculateTimeEstimate(gasLimit, gasPrice),
      });
    };
    
    estimate();
  }, [txRequest]);

  return (
    <div className="gas-feedback">
      {estimation ? (
        <>
          <p>Estimated gas: {ethers.utils.formatUnits(estimation.gasLimit.mul(estimation.gasPrice), 'gwei')} GWEI</p>
          <p>Typical confirmation: ~{estimation.timeEstimate} minutes</p>
        </>
      ) : (
        <Spinner size={14} />
      )}
    </div>
  );
};
```

## 4. Transaction Lifecycle Management

### Comprehensive Transaction Handler
```typescript
// hooks/useTransactionSender.ts
const useTransactionSender = () => {
  const { addTransaction, updateTransaction } = useTransactions();

  const sendTransaction = async (txRequest: ethers.providers.TransactionRequest, action?: string) => {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer = provider.getSigner();
    
    try {
      // Pre-transaction validation
      const estimatedGas = await provider.estimateGas(txRequest);
      const gasPrice = await provider.getGasPrice();
      
      // Add to transaction queue
      const txResponse = await signer.sendTransaction({
        ...txRequest,
        gasLimit: estimatedGas.mul(110).div(100), // 10% buffer
        gasPrice: gasPrice.mul(120).div(100), // 20% premium
      });
      
      const tx: Transaction = {
        hash: txResponse.hash,
        status: 'pending',
        message: getActionMessage(action),
        timestamp: Date.now(),
        chainId: (await provider.getNetwork()).chainId,
        action,
      };
      
      addTransaction(tx);
      
      // Wait for confirmation
      const receipt = await txResponse.wait();
      
      updateTransaction(tx.hash, {
        status: receipt.status === 1 ? 'confirmed' : 'failed',
        message: receipt.status === 1 
          ? `${action} completed successfully` 
          : `${action} failed`,
      });
      
      return receipt;
    } catch (error) {
      const message = error.message.includes('user rejected')
        ? 'Transaction rejected by user'
        : `Transaction failed: ${error.message.split('(')[0]}`;
        
      updateTransaction(txResponse.hash, {
        status: 'failed',
        message,
      });
      
      throw error;
    }
  };
  
  return { sendTransaction };
};
```

## 5. Enhanced Loading States

### Action-Specific Loading Components
```tsx
// components/ActionButton.tsx
const ActionButton = ({ action, onClick, children }) => {
  const { transactions } = useTransactions();
  const pendingTx = transactions.find(tx => 
    tx.action === action && tx.status === 'pending'
  );
  
  return (
    <button 
      onClick={onClick}
      disabled={!!pendingTx}
      className={pendingTx ? 'loading' : ''}
    >
      {pendingTx ? (
        <>
          <Spinner size={16} />
          Processing...
        </>
      ) : (
        children
      )}
      {pendingTx && (
        <Tooltip content={`Transaction pending: ${pendingTx.hash.slice(0, 8)}...`} />
      )}
    </button>
  );
};
```

### Chain-Specific Waiting Times
```typescript
// utils/networkUtils.ts
export const getAverageConfirmationTime = (chainId: number): number => {
  const times: Record<number, number> = {
    1: 1.5,    // Ethereum mainnet (minutes)
    137: 0.2,  // Polygon
    56: 0.3,   // BSC
    42161: 0.4 // Arbitrum
  };
  
  return times[chainId] || 1;
};

export const ProgressBar = ({ estimatedTime }: { estimatedTime: number }) => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 100 / (estimatedTime * 60 * 10), 100));
    }, 100);
    
    return () => clearInterval(interval);
  }, [estimatedTime]);
  
  return (
    <div className="progress-bar">
      <div style={{ width: `${progress}%` }} />
      <span>{Math.round(progress)}%</span>
    </div>
  );
};
```

## 6. Error Handling & Recovery

### Comprehensive Error States
```tsx
// components/TransactionError.tsx
const TransactionError = ({ error, onRetry }) => {
  const errorMessages = {
    'user rejected': 'You rejected the transaction',
    'gas too low': 'Insufficient gas - try increasing gas limit',
    'nonce too low': 'Pending transaction conflict - try again shortly',
    default: 'Transaction failed'
  };
  
  const message = Object.entries(errorMessages).find(([key]) => 
    error.message.toLowerCase().includes(key)
  )?.[1] || errorMessages.default;
  
  return (
    <div className="error-message">
      <AlertCircle size={18} />
      <span>{message}</span>
      <button onClick={onRetry}>Retry</button>
      <button onClick={() => copyToClipboard(error.message)}>Copy Error</button>
    </div>
  );
};
```

## Implementation Checklist

1. [ ] Set up centralized transaction state management
2. [ ] Create toast notification system with chain explorer links
3. [ ] Implement accurate gas estimation feedback
4. [ ] Design action-specific loading states
5. [ ] Add chain-aware confirmation time estimates
6. [ ] Build comprehensive error handling with recovery options
7. [ ] Develop persistent transaction history
8. [ ] Optimize for mobile transaction flows
9. [ ] Add transaction batching indicators
10. [ ] Implement cross-tab synchronization

These patterns create a seamless experience that keeps users informed throughout the entire transaction lifecycle while maintaining the decentralized nature of blockchain interactions. Would you like me to elaborate on any specific aspect of the transaction UX design?