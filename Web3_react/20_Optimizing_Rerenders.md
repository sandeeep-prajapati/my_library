# Optimizing React Performance for Blockchain Data Polling

Polling blockchain data efficiently is crucial for dApp performance. Here's a comprehensive approach to optimize your React components:

## 1. Smart Polling Strategies

### Adaptive Polling with Exponential Backoff
```typescript
// hooks/useBlockchainPolling.ts
import { useEffect, useRef } from 'react';
import { ethers } from 'ethers';

export function useBlockchainPolling(
  callback: () => Promise<void>,
  dependencies: any[] = [],
  baseInterval = 5000,
  maxInterval = 30000
) {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const currentInterval = useRef(baseInterval);
  const isMounted = useRef(true);

  const poll = async () => {
    if (!isMounted.current) return;
    
    try {
      await callback();
      // Reset to base interval on success
      currentInterval.current = baseInterval;
    } catch (error) {
      // Exponential backoff on error
      currentInterval.current = Math.min(
        currentInterval.current * 2,
        maxInterval
      );
    } finally {
      if (isMounted.current) {
        timeoutRef.current = setTimeout(poll, currentInterval.current);
      }
    }
  };

  useEffect(() => {
    isMounted.current = true;
    poll();
    
    return () => {
      isMounted.current = false;
      clearTimeout(timeoutRef.current);
    };
  }, dependencies);
}
```

## 2. Efficient Data Fetching

### Batch RPC Requests
```typescript
// utils/blockchain.ts
export async function fetchMultipleBalances(
  provider: ethers.providers.Provider,
  tokens: string[],
  address: string
) {
  const batch = new ethers.providers.JsonRpcBatchProvider(
    provider.connection.url
  );

  const balanceCalls = tokens.map(token => {
    const contract = new ethers.Contract(
      token,
      ['function balanceOf(address) view returns (uint256)'],
      batch
    );
    return contract.balanceOf(address);
  });

  return Promise.all(balanceCalls);
}
```

### Memoized Selectors
```typescript
// selectors/blockchain.ts
import { createSelector } from 'reselect';

const selectRawBlockchainData = (state) => state.blockchain;

export const selectFormattedBalances = createSelector(
  [selectRawBlockchainData],
  (blockchain) => {
    return Object.entries(blockchain.balances).map(([token, balance]) => ({
      token,
      balance: ethers.utils.formatUnits(balance, 18),
    }));
  }
);
```

## 3. React Optimization Techniques

### Virtualized Lists for Large Data
```tsx
// components/TokenList.tsx
import { FixedSizeList as List } from 'react-window';

const TokenList = ({ tokens }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <TokenItem token={tokens[index]} />
    </div>
  );

  return (
    <List
      height={500}
      itemCount={tokens.length}
      itemSize={60}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

### Debounced Updates
```typescript
// hooks/useDebouncedPolling.ts
import { useEffect, useState } from 'react';
import { debounce } from 'lodash';

export function useDebouncedPolling(
  fetchFn: () => Promise<any>,
  interval: number,
  debounceTime = 300
) {
  const [data, setData] = useState<any>(null);
  
  useEffect(() => {
    const debouncedSetData = debounce(setData, debounceTime);
    const fetchData = async () => {
      const result = await fetchFn();
      debouncedSetData(result);
    };
    
    fetchData();
    const intervalId = setInterval(fetchData, interval);
    
    return () => {
      clearInterval(intervalId);
      debouncedSetData.cancel();
    };
  }, [fetchFn, interval, debounceTime]);
  
  return data;
}
```

## 4. WebSocket Subscriptions (Alternative to Polling)

```typescript
// hooks/useWebSocketUpdates.ts
import { useEffect, useState } from 'react';
import { ethers } from 'ethers';

export function useWebSocketUpdates(
  contractAddress: string,
  abi: ethers.ContractInterface,
  eventName: string
) {
  const [data, setData] = useState<any>(null);
  
  useEffect(() => {
    const provider = new ethers.providers.WebSocketProvider(
      process.env.INFURA_WSS_URL
    );
    const contract = new ethers.Contract(contractAddress, abi, provider);
    
    contract.on(eventName, (...args) => {
      setData(args);
    });
    
    return () => {
      contract.removeAllListeners();
      provider._websocket.close();
    };
  }, [contractAddress, abi, eventName]);
  
  return data;
}
```

## 5. Performance Monitoring

### Custom Profiling Hook
```typescript
// hooks/useRenderProfiler.ts
import { useEffect, useRef } from 'react';

export function useRenderProfiler(name: string) {
  const renderCount = useRef(0);
  const lastRenderTime = useRef(performance.now());

  useEffect(() => {
    renderCount.current += 1;
    const now = performance.now();
    const renderTime = now - lastRenderTime.current;
    
    if (renderTime > 16) { // 60fps threshold
      console.warn(`Slow render (${renderTime.toFixed(2)}ms) in ${name}`);
    }
    
    lastRenderTime.current = now;
  });
}
```

## 6. Optimized Context Updates

### Split Contexts for Independent Updates
```tsx
// contexts/BlockchainContext.tsx
import { createContext, useContext } from 'react';

const BalanceContext = createContext<Record<string, ethers.BigNumber>>({});
const TransactionContext = createContext<Transaction[]>([]);

export const BlockchainProvider = ({ children }) => {
  const [balances, setBalances] = useState({});
  const [transactions, setTransactions] = useState([]);
  
  // Update logic here
  
  return (
    <BalanceContext.Provider value={balances}>
      <TransactionContext.Provider value={transactions}>
        {children}
      </TransactionContext.Provider>
    </BalanceContext.Provider>
  );
};

export const useBalances = () => useContext(BalanceContext);
export const useTransactions = () => useContext(TransactionContext);
```

## Implementation Checklist

1. [ ] Replace simple polling with adaptive strategies
2. [ ] Batch RPC calls where possible
3. [ ] Implement memoization for derived data
4. [ ] Use virtualization for large lists
5. [ ] Add debouncing for rapid updates
6. [ ] Consider WebSocket subscriptions for real-time data
7. [ ] Profile component renders
8. [ ] Split contexts to minimize unnecessary updates
9. [ ] Optimize re-renders with React.memo
10. [ ] Implement loading states and error boundaries

## Key Optimization Patterns

- **Request Deduplication**: Cache identical requests
- **Data Normalization**: Store blockchain data in flat structures
- **Selective Updates**: Only fetch changed data
- **Priority Loading**: Load critical data first
- **Background Sync**: Update non-critical data in background

```typescript
// Example of request deduplication
const pendingRequests = new Map();

async function fetchWithDedupe(key: string, fetchFn: () => Promise<any>) {
  if (pendingRequests.has(key)) {
    return pendingRequests.get(key);
  }
  
  const promise = fetchFn();
  pendingRequests.set(key, promise);
  
  try {
    return await promise;
  } finally {
    pendingRequests.delete(key);
  }
}
```