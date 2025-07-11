# Deploying Web3 dApps to Decentralized Hosting

Deploying a React + Web3 application to decentralized platforms requires special considerations for blockchain compatibility. Here's how to deploy to IPFS, Fleek, and Vercel:

## 1. Preparing Your dApp for Decentralized Hosting

### Essential Configurations:
```javascript
// vite.config.js or webpack.config.js
export default {
  base: './', // Relative paths for IPFS
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  // Enable hash routing for SPAs
  server: {
    historyApiFallback: true,
  }
}
```

### Environment Variables:
```javascript
// src/config.js
export const CONFIG = {
  INFURA_ID: import.meta.env.VITE_INFURA_ID || 'fallback-project-id',
  CONTRACT_ADDRESSES: {
    1: '0x...', // Mainnet
    5: '0x...', // Goerli
    137: '0x...' // Polygon
  }
}
```

## 2. Deploying to IPFS (Manual)

### Build and Deploy:
```bash
# Build production bundle
npm run build

# Install IPFS CLI
npm install -g ipfs

# Add to IPFS
ipfs add -r dist/

# Pin to your node
ipfs pin add -r <CID>
```

### IPFS-Specific Code Adjustments:
```javascript
// Detect IPFS gateway
const getGateway = () => {
  if (window.location.host.includes('ipfs') || window.location.host.includes('ipns')) {
    return `${window.location.protocol}//${window.location.host}`;
  }
  return 'https://ipfs.io/ipfs/';
};
```

## 3. Deploying with Fleek (Automated IPFS)

### `.fleek.json` Configuration:
```json
{
  "build": {
    "image": "node:18",
    "command": "npm install && npm run build",
    "publicDir": "dist",
    "baseDir": "."
  },
  "routes": [
    {
      "path": "/*",
      "handle": "filesystem"
    }
  ]
}
```

### Deployment Steps:
1. Connect your GitHub/GitLab repo to Fleek
2. Select "Host on IPFS" option
3. Fleek will automatically:
   - Build your app
   - Generate IPFS hash
   - Create IPNS pointer for mutable content
   - Optionally set up ENS domain (e.g., myapp.eth)

## 4. Deploying to Vercel (Hybrid Approach)

### `vercel.json` for Web3 Compatibility:
```json
{
  "version": 2,
  "rewrites": [
    {
      "source": "/.well-known/(.*)",
      "destination": "/api/well-known/$1"
    },
    {
      "source": "/manifest.json",
      "destination": "/api/manifest"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Content-Security-Policy", 
          "value": "default-src 'self' https:; connect-src 'self' https://*.infura.io https://*.alchemyapi.io" }
      ]
    }
  ]
}
```

### Web3-Specific Vercel Settings:
1. Enable "Include source files" in build settings
2. Add environment variables for:
   - Alchemy/Infura IDs
   - Contract addresses per network
3. Configure CORS headers properly

## 5. Multi-Gateway Fallback System

```javascript
// src/utils/web3Fallbacks.js
export const IPFS_GATEWAYS = [
  'https://cloudflare-ipfs.com/ipfs/',
  'https://ipfs.io/ipfs/',
  'https://dweb.link/ipfs/'
];

export const loadFromGateway = async (cid, path = '') => {
  for (const gateway of IPFS_GATEWAYS) {
    try {
      const res = await fetch(`${gateway}${cid}${path}`);
      if (res.ok) return res;
    } catch (e) {
      console.warn(`Failed ${gateway}`, e);
    }
  }
  throw new Error('All gateways failed');
};
```

## 6. ENS + IPFS Integration

1. Upload build to IPFS (get CID)
2. Set contenthash in your ENS domain:
```javascript
// Using ethers.js
const resolver = await provider.getResolver('myapp.eth');
await resolver.setContentHash('ipfs://<CID>');
```

3. Access via:
- `https://myapp.eth.limo` (HTTPS gateway)
- `ipfs://<CID>` (Native IPFS)
- `https://cloudflare-ipfs.com/ipns/myapp.eth`

## 7. Continuous Deployment Workflow

### GitHub Actions for Fleek/Vercel:
```yaml
name: Deploy dApp
on: [push]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install
        run: npm install
        
      - name: Build
        run: npm run build
        
      - name: Deploy to IPFS
        uses: fleekhq/fleek-action@v0.1
        with:
          api-key: ${{ secrets.FLEEK_API_KEY }}
          team-id: ${{ secrets.FLEEK_TEAM_ID }}
          
      - name: Update ENS
        run: |
          npm run update-ens --cid=${{ steps.ipfs.outputs.cid }}
```

## Critical Web3 Hosting Considerations

1. **Dynamic Configuration**: 
   - Store contract addresses in decentralized storage (IPFS/Arweave)
   - Use environment variables during build

2. **Routing Solutions**:
```javascript
// HashRouter for IPFS compatibility
import { HashRouter } from 'react-router-dom';

ReactDOM.render(
  <HashRouter>
    <App />
  </HashRouter>,
  document.getElementById('root')
);
```

3. **Cache Busting**:
```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        entryFileNames: `[name].[hash].js`,
        chunkFileNames: `[name].[hash].js`,
        assetFileNames: `[name].[hash].[ext]`
      }
    }
  }
}
```

4. **Service Worker Offline Support**:
```javascript
// src/service-worker.js
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('infura.io')) {
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request);
      })
    );
  }
});
```

## Deployment Checklist

1. [ ] Test build with `npm run build` locally
2. [ ] Verify all assets load with relative paths
3. [ ] Ensure Web3 providers have proper CORS headers
4. [ ] Configure redirects for SPA routing
5. [ ] Set up proper CSP headers for security
6. [ ] Implement gateway fallback system
7. [ ] Configure ENS domain if applicable
8. [ ] Set up CI/CD pipeline
