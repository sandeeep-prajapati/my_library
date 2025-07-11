# Testing React Components with Smart Contract Interactions

Testing components that interact with blockchain contracts requires mocking Web3 providers and contract responses. Here's a comprehensive approach using Jest and ethers.js mock providers:

## 1. Setup Mock Web3 Provider

Create a reusable mock provider utility:

```typescript
// test-utils/mockWeb3.ts
import { ethers } from 'ethers';

export class MockWeb3Provider extends ethers.providers.Web3Provider {
  constructor() {
    super({
      request: jest.fn(),
    } as any);
  }

  mockRequestResolve(result: any) {
    (this.provider.request as jest.Mock).mockResolvedValueOnce(result);
  }

  mockRequestReject(error: any) {
    (this.provider.request as jest.Mock).mockRejectedValueOnce(error);
  }
}

export const mockAccount = '0x71C7656EC7ab88b098defB751B7401B5f6d8976F';
export const mockChainId = 5; // Goerli
```

## 2. Mock Contract Setup

```typescript
// test-utils/mockContract.ts
import { Contract } from 'ethers';

export const mockContract = {
  connect: jest.fn().mockReturnThis(),
  functions: {
    balanceOf: jest.fn(),
    transfer: jest.fn(),
  },
  filters: {
    Transfer: jest.fn(),
  },
  queryFilter: jest.fn(),
  estimateGas: {
    transfer: jest.fn(),
  },
} as unknown as Contract;
```

## 3. Component Test Example

Test a component that displays token balance:

```typescript
// components/TokenBalance.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { MockWeb3Provider, mockAccount } from '../test-utils/mockWeb3';
import { mockContract } from '../test-utils/mockContract';
import TokenBalance from './TokenBalance';

jest.mock('../hooks/useContract', () => ({
  __esModule: true,
  default: () => mockContract,
}));

describe('TokenBalance', () => {
  let mockProvider: MockWeb3Provider;

  beforeEach(() => {
    mockProvider = new MockWeb3Provider();
    mockContract.functions.balanceOf.mockResolvedValue([ethers.BigNumber.from('1000000000000000000')]); // 1 token
  });

  it('displays token balance when connected', async () => {
    render(
      <Web3Context.Provider value={{
        provider: mockProvider,
        account: mockAccount,
        chainId: 5,
        isConnected: true,
        connectWallet: jest.fn(),
        disconnectWallet: jest.fn(),
      }}>
        <TokenBalance tokenAddress="0x123..." />
      </Web3Context.Provider>
    );

    await waitFor(() => {
      expect(screen.getByText('Balance: 1.0')).toBeInTheDocument();
    });
  });

  it('shows connect button when not connected', () => {
    render(
      <Web3Context.Provider value={{
        provider: undefined,
        account: '',
        chainId: 0,
        isConnected: false,
        connectWallet: jest.fn(),
        disconnectWallet: jest.fn(),
      }}>
        <TokenBalance tokenAddress="0x123..." />
      </Web3Context.Provider>
    );

    expect(screen.getByText('Connect to view balance')).toBeInTheDocument();
  });
});
```

## 4. Testing Contract Interactions

Test a component that sends transactions:

```typescript
// components/SendTransaction.test.tsx
describe('SendTransaction', () => {
  it('handles successful transaction', async () => {
    mockContract.estimateGas.transfer.mockResolvedValue(ethers.BigNumber.from('50000'));
    mockContract.functions.transfer.mockResolvedValue({
      wait: jest.fn().mockResolvedValue({
        status: 1,
        transactionHash: '0x123...',
      }),
    });

    render(
      <Web3Context.Provider value={mockConnectedContext}>
        <SendTransaction />
      </Web3Context.Provider>
    );

    fireEvent.click(screen.getByText('Send 1 ETH'));
    
    await waitFor(() => {
      expect(screen.getByText('Transaction sent!')).toBeInTheDocument();
    });
  });

  it('handles transaction rejection', async () => {
    mockContract.estimateGas.transfer.mockRejectedValue(new Error('user rejected'));

    render(
      <Web3Context.Provider value={mockConnectedContext}>
        <SendTransaction />
      </Web3Context.Provider>
    );

    fireEvent.click(screen.getByText('Send 1 ETH'));
    
    await waitFor(() => {
      expect(screen.getByText('Transaction rejected')).toBeInTheDocument();
    });
  });
});
```

## 5. Advanced Testing Scenarios

### Testing Event Listeners

```typescript
it('updates balance on Transfer events', async () => {
  const mockEventListener = jest.fn();
  mockContract.on.mockImplementation((event, callback) => {
    if (event === 'Transfer') {
      mockEventListener.mockImplementation(() => callback(
        mockAccount, 
        '0xrecipient', 
        ethers.BigNumber.from('100')
      ));
    }
    return { remove: jest.fn() };
  });

  render(
    <Web3Context.Provider value={mockConnectedContext}>
      <TokenBalance />
    </Web3Context.Provider>
  );

  act(() => {
    mockEventListener(); // Trigger mock event
  });

  await waitFor(() => {
    expect(screen.getByText('Balance: 1.0')).toBeInTheDocument();
  });
});
```

### Testing Chain Changes

```typescript
it('refetches data on chain change', async () => {
  const mockConnect = jest.fn();
  const contextValue = {
    ...mockConnectedContext,
    chainId: 1, // Initially Mainnet
    provider: mockProvider,
  };

  const { rerender } = render(
    <Web3Context.Provider value={contextValue}>
      <ChainAwareComponent />
    </Web3Context.Provider>
  );

  // Change to Polygon
  rerender(
    <Web3Context.Provider value={{ ...contextValue, chainId: 137 }}>
      <ChainAwareComponent />
    </Web3Context.Provider>
  );

  await waitFor(() => {
    expect(mockContract.functions.balanceOf).toHaveBeenCalledTimes(2);
  });
});
```

## 6. Test Utility Functions

Create helper functions for common test setups:

```typescript
// test-utils/web3Helpers.ts
export const getMockWeb3Context = (overrides = {}) => ({
  provider: new MockWeb3Provider(),
  account: mockAccount,
  chainId: mockChainId,
  isConnected: true,
  connectWallet: jest.fn(),
  disconnectWallet: jest.fn(),
  error: undefined,
  ...overrides,
});

export const renderWithWeb3 = (ui: React.ReactElement, contextOverrides = {}) => {
  return render(
    <Web3Context.Provider value={getMockWeb3Context(contextOverrides)}>
      {ui}
    </Web3Context.Provider>
  );
};
```

## Best Practices

1. **Isolate Tests**: Don't rely on previous test state
2. **Mock All External Dependencies**: Especially blockchain interactions
3. **Test Error States**: Rejections, wrong network, etc.
4. **Use TypeScript**: Catch type errors during development
5. **Test Loading States**: Components should handle async operations gracefully
6. **Clean Up Event Listeners**: Verify cleanup in afterEach hooks

```typescript
afterEach(() => {
  jest.clearAllMocks();
  mockContract.on.mockClear();
  (mockContract.on as jest.Mock).mockReturnValue({ remove: jest.fn() });
});
```

This testing approach gives you:
- Fast execution (no real blockchain needed)
- Deterministic results
- Full control over test scenarios
- Complete coverage of success/error cases
- Type safety with TypeScript
