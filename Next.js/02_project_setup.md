---

### 1. **Create a New Next.js Project**

First, use the `create-next-app` command with TypeScript template options.

```bash
npx create-next-app@latest my-nextjs-app --typescript
```

Alternatively, with `yarn` or `pnpm`:

```bash
yarn create next-app my-nextjs-app --typescript
```

This sets up a new Next.js project with TypeScript configuration out of the box, installing all the necessary dependencies.

---

### 2. **Project Structure Overview**

The basic structure of a Next.js project created with TypeScript will look like this:

```plaintext
my-nextjs-app/
├── public/
│   └── favicon.ico           # Favicon and other public assets
├── src/
│   ├── components/           # Reusable components
│   ├── pages/                # Next.js pages
│   │   ├── api/              # API routes (Node.js backend code)
│   │   ├── _app.tsx          # Custom App component
│   │   ├── _document.tsx     # Custom Document component
│   │   └── index.tsx         # Main page component
│   ├── styles/               # CSS/SCSS modules or global styles
│   │   ├── globals.css       # Global CSS file
│   │   └── Home.module.css   # Component-scoped CSS module
│   └── types/                # TypeScript type definitions (optional)
├── .eslintrc.json            # ESLint configuration file
├── .gitignore                # Files and directories to ignore in Git
├── next-env.d.ts             # TypeScript definitions for Next.js
├── next.config.js            # Next.js configuration
├── package.json              # Project metadata and dependencies
├── tsconfig.json             # TypeScript configuration
└── README.md                 # Project documentation
```

---

### 3. **Configure TypeScript Settings**

- **`tsconfig.json`**: This file will be auto-generated and contains the TypeScript configuration. You can modify it to customize your TypeScript settings. For example:

  ```json
  {
    "compilerOptions": {
      "target": "es5",
      "lib": ["dom", "dom.iterable", "esnext"],
      "allowJs": true,
      "skipLibCheck": true,
      "strict": true,
      "forceConsistentCasingInFileNames": true,
      "noEmit": true,
      "incremental": true,
      "jsx": "preserve",
      "moduleResolution": "node",
      "resolveJsonModule": true,
      "isolatedModules": true,
      "esModuleInterop": true,
      "module": "esnext",
      "baseUrl": "./",
      "paths": {
        "@/components/*": ["src/components/*"],
        "@/pages/*": ["src/pages/*"]
      }
    },
    "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
    "exclude": ["node_modules"]
  }
  ```

  Here, we set strict type-checking, enable module resolution, and define paths for cleaner imports using `@/components/` and `@/pages/`.

### 4. **Add Basic Pages and Components**

#### **Pages**
- **`src/pages/index.tsx`**: Main entry point of the application.

  ```tsx
  import type { NextPage } from 'next';
  import Head from 'next/head';
  import styles from '@/styles/Home.module.css';

  const Home: NextPage = () => {
    return (
      <div className={styles.container}>
        <Head>
          <title>My Next.js App</title>
          <meta name="description" content="Generated by create next app" />
          <link rel="icon" href="/favicon.ico" />
        </Head>

        <main className={styles.main}>
          <h1>Welcome to My Next.js App with TypeScript!</h1>
        </main>
      </div>
    );
  };

  export default Home;
  ```

#### **Components**
- **`src/components/Header.tsx`**: A simple header component.

  ```tsx
  import React from 'react';

  const Header: React.FC = () => {
    return (
      <header>
        <h1>My Header</h1>
      </header>
    );
  };

  export default Header;
  ```

#### **Styles**
- **`src/styles/globals.css`**: Global styles for your application.
- **`src/styles/Home.module.css`**: Component-scoped CSS for the `Home` page.

### 5. **Add Type Definitions (Optional)**

To improve code readability and type safety, create custom type definitions in a `src/types` directory:

- **Example: `src/types/index.ts`**

  ```typescript
  export interface User {
    id: number;
    name: string;
    email: string;
  }
  ```

You can then import these types into any component or page where they’re needed.

### 6. **Set Up Absolute Imports**

In `tsconfig.json`, ensure `paths` are set for easy import statements (as shown in Step 3). Now, you can import components and other files like this:

```typescript
import Header from '@/components/Header';
```

### 7. **Run the Application**

Finally, to start the development server, run:

```bash
npm run dev
```

Visit `http://localhost:3000` in your browser to see the Next.js app with TypeScript in action.

---

