Installing MongoDB varies depending on the platform you are using (Windows, macOS, or Linux). Below are the installation instructions for each platform along with the necessary configuration steps.

### 1. Installing MongoDB on Windows

#### Installation Steps:
1. **Download MongoDB**:
   - Go to the [MongoDB Download Center](https://www.mongodb.com/try/download/community).
   - Select the version you want and choose "Windows" as the operating system.
   - Download the `.msi` installer.

2. **Run the Installer**:
   - Double-click the downloaded `.msi` file.
   - Choose "Complete" setup when prompted.

3. **Install MongoDB as a Service**:
   - During installation, you can choose to install MongoDB as a Windows Service. This will allow it to run in the background automatically.
   - Configure the service to run with the default options.

4. **Set up the Data Directory**:
   - MongoDB requires a data directory to store its data. By default, it uses `C:\data\db`.
   - Create the directory if it doesn’t exist:
     ```bash
     mkdir C:\data\db
     ```

5. **Add MongoDB to the System Path**:
   - Add the MongoDB `bin` folder (e.g., `C:\Program Files\MongoDB\Server\<version>\bin`) to your system `PATH` environment variable.

#### Configuration Steps:
1. **Start MongoDB**:
   - Open a Command Prompt and run:
     ```bash
     mongod
     ```
   - This command starts the MongoDB server.

2. **Connect to MongoDB**:
   - Open another Command Prompt and run:
     ```bash
     mongo
     ```
   - This command connects you to the MongoDB shell.

### 2. Installing MongoDB on macOS

#### Installation Steps:
1. **Install Homebrew (if not already installed)**:
   - Open a Terminal and run:
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```

2. **Tap the MongoDB Formula**:
   - Run the following command in the Terminal:
     ```bash
     brew tap mongodb/brew
     ```

3. **Install MongoDB**:
   - Install MongoDB using Homebrew:
     ```bash
     brew install mongodb-community@<version>
     ```

#### Configuration Steps:
1. **Start MongoDB**:
   - You can start MongoDB as a service using:
     ```bash
     brew services start mongodb/brew/mongodb-community
     ```

2. **Connect to MongoDB**:
   - In the Terminal, run:
     ```bash
     mongo
     ```
   - This connects you to the MongoDB shell.

### 3. Installing MongoDB on Linux (Ubuntu)

#### Installation Steps:
1. **Import the MongoDB Public Key**:
   - Open a Terminal and run:
     ```bash
     wget -qO - https://www.mongodb.org/static/pgp/server-<version>.asc | sudo apt-key add -
     ```

2. **Create the List File**:
   - Create a MongoDB list file in `/etc/apt/sources.list.d/mongodb-org-<version>.list`:
     ```bash
     echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/multiverse amd64 mongodb-org <version> multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-<version>.list
     ```

3. **Update the Package Database**:
   - Run:
     ```bash
     sudo apt-get update
     ```

4. **Install MongoDB**:
   - Install MongoDB with:
     ```bash
     sudo apt-get install -y mongodb-org
     ```

#### Configuration Steps:
1. **Start MongoDB**:
   - Start the MongoDB service:
     ```bash
     sudo systemctl start mongod
     ```
   - Optionally, enable it to start on boot:
     ```bash
     sudo systemctl enable mongod
     ```

2. **Check the Status**:
   - Verify that MongoDB is running:
     ```bash
     sudo systemctl status mongod
     ```

3. **Connect to MongoDB**:
   - Run the MongoDB shell:
     ```bash
     mongo
     ```

### Additional Configuration

Regardless of the platform, you may want to perform additional configurations:

- **Configure `mongod.conf`**:
  - The configuration file is typically located at `/etc/mongod.conf` on Linux and `C:\Program Files\MongoDB\Server\<version>\bin\mongod.cfg` on Windows.
  - You can configure various options such as the storage engine, logging, network interfaces, and security.

- **Data Directory Permissions** (Linux):
  - Ensure that the MongoDB user has appropriate permissions on the data directory:
    ```bash
    sudo chown -R mongodb:mongodb /var/lib/mongodb
    sudo chown -R mongodb:mongodb /var/log/mongodb
    ```

- **Network Binding**:
  - To allow remote access, you may need to modify the `bindIp` setting in `mongod.conf` to `0.0.0.0` or specify the server’s IP address. Be cautious about security implications when allowing remote access.

### Conclusion

After following the installation and configuration steps outlined above, you should have a functioning MongoDB setup on your respective platform. Ensure you consult the official MongoDB documentation for more detailed instructions and configurations, as well as best practices for securing and optimizing your MongoDB installation.