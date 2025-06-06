Here’s a guide on creating a Bash script to install Next.js across different platforms (Linux, macOS, and Windows with WSL). This script will automate the installation of Node.js (if needed), Next.js, and other dependencies to get a Next.js project running smoothly.

---

### 1. **Prerequisites**
   - Ensure you have Bash installed. (Bash comes pre-installed on macOS and most Linux distributions. Windows users can install Bash via WSL.)

---

### 2. **Create the Script**
   - Open a terminal.
   - Use your preferred editor to create the script file:
     ```bash
     nano install_nextjs.sh
     ```

---

### 3. **Bash Script for Installing Next.js**

   Here's the script that covers installation for **Linux**, **macOS**, and **Windows (using WSL)**.

   ```bash
   #!/bin/bash

   # Check for sudo/root privileges
   if [[ $EUID -ne 0 ]]; then
      echo "This script must be run as root or with sudo privileges" 
      exit 1
   fi

   # Function to install Node.js and npm
   install_node() {
      echo "Installing Node.js and npm..."
      if [[ "$OSTYPE" == "linux-gnu"* ]]; then
         curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
         sudo apt install -y nodejs
      elif [[ "$OSTYPE" == "darwin"* ]]; then
         brew install node
      else
         echo "Unsupported OS type: $OSTYPE. Install Node.js manually."
         exit 1
      fi
      echo "Node.js and npm installed successfully!"
   }

   # Function to check if Node.js is installed
   check_node() {
      if ! command -v node &> /dev/null
      then
         echo "Node.js could not be found"
         install_node
      else
         echo "Node.js is already installed"
      fi
   }

   # Function to install Next.js
   install_nextjs() {
      echo "Installing Next.js..."
      npx create-next-app@latest my-nextjs-app
      echo "Next.js installed successfully!"
      echo "To start the Next.js development server, run:"
      echo "cd my-nextjs-app && npm run dev"
   }

   # Detect OS type and proceed with installation
   echo "Detecting your OS..."
   case "$OSTYPE" in
      linux-gnu*)
         echo "Linux OS detected"
         sudo apt update
         sudo apt install -y curl build-essential
         check_node
         install_nextjs
         ;;
      darwin*)
         echo "macOS detected"
         if ! command -v brew &> /dev/null; then
            echo "Homebrew is required but not installed. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
         fi
         check_node
         install_nextjs
         ;;
      msys*|cygwin*|win32*)
         echo "Windows Subsystem for Linux (WSL) detected"
         sudo apt update
         sudo apt install -y curl build-essential
         check_node
         install_nextjs
         ;;
      *)
         echo "Unsupported OS. Exiting."
         exit 1
         ;;
   esac

   echo "Installation complete!"
   ```

---

### 4. **Running the Script**

1. Save and close the script.
2. Make the script executable by running:
   ```bash
   chmod +x install_nextjs.sh
   ```
3. Run the script with sudo privileges:
   ```bash
   sudo ./install_nextjs.sh
   ```

---

### 5. **Script Breakdown**

- **`install_node` Function**: Checks the OS type and installs Node.js and npm using either `apt` (for Linux) or `brew` (for macOS). This section can be extended to cover more package managers if needed.

- **`check_node` Function**: Checks if Node.js is already installed. If it’s not found, it triggers `install_node`.

- **`install_nextjs` Function**: Uses `npx` to create a new Next.js application in a directory named `my-nextjs-app`.

- **OS Detection and Installation**: Uses `$OSTYPE` to identify the OS and runs the appropriate commands for installing dependencies and Next.js.

---

### 6. **Output After Installation**
After running the script:
- Next.js will be installed in the `my-nextjs-app` directory.
- Instructions for starting the development server will be provided in the terminal.

This script is a one-stop setup solution for anyone needing Next.js installed on Linux, macOS, or WSL.