### Node.js Installation and Development Environment Setup

---

## Overview
Node.js is a JavaScript runtime built on Chrome's V8 engine, allowing developers to build scalable network applications. This guide outlines the steps to install Node.js and set up a development environment on various operating systems.

### Installation Steps

#### 1. **Windows Installation**
   - **Download Node.js:**
     - Go to the [Node.js official website](https://nodejs.org/).
     - Download the Windows Installer (LTS version recommended).
   
   - **Run the Installer:**
     - Open the downloaded `.msi` file.
     - Follow the installation prompts and accept the license agreement.
     - Select the destination folder and components to install (Node.js and npm are selected by default).
     - Click "Install" to complete the installation.

   - **Verify Installation:**
     - Open Command Prompt and run:
       ```bash
       node -v
       npm -v
       ```
     - This should display the installed versions of Node.js and npm.

#### 2. **macOS Installation**
   - **Using Homebrew:**
     - Open Terminal.
     - If Homebrew is not installed, you can install it using:
       ```bash
       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
       ```
     - Install Node.js via Homebrew:
       ```bash
       brew install node
       ```

   - **Verify Installation:**
     - Check the installed versions:
       ```bash
       node -v
       npm -v
       ```

#### 3. **Linux Installation**
   - **Debian/Ubuntu:**
     - Open Terminal and update your package index:
       ```bash
       sudo apt update
       ```
     - Install Node.js using the following command:
       ```bash
       sudo apt install nodejs npm
       ```

   - **CentOS/RHEL:**
     - Use the NodeSource repository for the latest version:
       ```bash
       curl -sL https://rpm.nodesource.com/setup_16.x | sudo bash -
       sudo yum install nodejs
       ```

   - **Verify Installation:**
     - Check the installed versions:
       ```bash
       node -v
       npm -v
       ```

### Setting Up a Development Environment

1. **Choose a Code Editor:**
   - Popular choices include [Visual Studio Code](https://code.visualstudio.com/), [Sublime Text](https://www.sublimetext.com/), and [Atom](https://atom.io/).
   - Download and install your preferred code editor.

2. **Create a New Project Directory:**
   - Open Terminal or Command Prompt and create a new directory:
     ```bash
     mkdir my-node-app
     cd my-node-app
     ```

3. **Initialize a New Node.js Project:**
   - Run the following command to create a `package.json` file:
     ```bash
     npm init -y
     ```
   - This command initializes a new Node.js project with default settings.

4. **Install Dependencies:**
   - If you need any libraries, such as Express.js for web applications, you can install them using npm:
     ```bash
     npm install express
     ```

5. **Create Your Main Application File:**
   - Create an `index.js` file in your project directory. This will be the main entry point for your application:
     ```bash
     touch index.js
     ```
   - Open `index.js` in your code editor and add the following sample code:
     ```javascript
     const express = require('express');
     const app = express();
     const PORT = process.env.PORT || 3000;

     app.get('/', (req, res) => {
         res.send('Hello, Node.js!');
     });

     app.listen(PORT, () => {
         console.log(`Server is running on http://localhost:${PORT}`);
     });
     ```

6. **Run Your Application:**
   - In your Terminal, run the following command to start your application:
     ```bash
     node index.js
     ```
   - Open your browser and go to `http://localhost:3000` to see your application running.

### Conclusion
You have successfully installed Node.js and set up a development environment. You can now start building your applications using JavaScript on the server side. For further development, explore Node.js libraries and frameworks like Express, Koa, and more.