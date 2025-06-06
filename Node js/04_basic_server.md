Creating a basic HTTP server using Node.js is straightforward and can be done in just a few steps. Below are the instructions for setting up a simple HTTP server that responds to requests.

### Step-by-Step Guide to Create a Basic HTTP Server

#### 1. **Install Node.js**
Ensure you have Node.js installed. You can check this by running:
```bash
node -v
```
If you don’t have it installed, you can download it from the [official Node.js website](https://nodejs.org/).

#### 2. **Create a New Project Directory**
Open your terminal and create a new directory for your project. Navigate into the directory:
```bash
mkdir my-http-server
cd my-http-server
```

#### 3. **Initialize a New Node.js Project**
Initialize a new Node.js project using npm:
```bash
npm init -y
```
This command creates a `package.json` file with default settings.

#### 4. **Create the Server File**
Create a new file called `server.js`:
```bash
touch server.js
```

#### 5. **Write the Server Code**
Open `server.js` in your favorite text editor and add the following code:

```javascript
// Load the http module to create an HTTP server.
const http = require('http');

// Configure the HTTP server to respond with a message.
const hostname = '127.0.0.1'; // Localhost
const port = 3000; // Port number

// Create the server
const server = http.createServer((req, res) => {
    res.statusCode = 200; // HTTP status code for success
    res.setHeader('Content-Type', 'text/plain'); // Set the content type
    res.end('Hello, World!\n'); // Response message
});

// Start the server and listen on the specified port
server.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`);
});
```

#### 6. **Run the Server**
In the terminal, run the following command to start your server:
```bash
node server.js
```
You should see the message:
```
Server running at http://127.0.0.1:3000/
```

#### 7. **Access the Server**
Open a web browser and go to `http://127.0.0.1:3000/`. You should see the message:
```
Hello, World!
```

#### 8. **Stopping the Server**
To stop the server, go back to your terminal and press `Ctrl + C`.

### Explanation of the Code
- **http Module**: This is a core module in Node.js that allows you to create HTTP servers.
- **createServer() Method**: This method creates a new HTTP server and takes a callback function that is called for every request to the server.
- **Response**: The `res` object is used to send a response back to the client. You can set the status code, headers, and the response body.
- **listen() Method**: This method tells the server to start listening for incoming connections on a specific hostname and port.

### Conclusion
You have now created a basic HTTP server using Node.js! From here, you can expand your server to handle different routes, serve static files, or integrate it with a web framework like Express for more complex applications.