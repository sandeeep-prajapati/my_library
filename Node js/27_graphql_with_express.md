Setting up and using GraphQL with Express.js involves several steps, including installing the necessary packages, defining a GraphQL schema, and implementing the server logic. Below is a step-by-step guide to help you get started.

### Step 1: Set Up Your Node.js and Express Project

1. **Create a New Directory for Your Project**

   ```bash
   mkdir graphql-express
   cd graphql-express
   ```

2. **Initialize a New Node.js Project**

   ```bash
   npm init -y
   ```

3. **Install Required Packages**

   You'll need the following packages:
   - `express` for setting up the server
   - `graphql` for defining the schema and queries
   - `express-graphql` to integrate GraphQL with Express

   Install them using:

   ```bash
   npm install express graphql express-graphql
   ```

### Step 2: Create a Basic Express Server with GraphQL

Create a new file named `server.js` and set up a basic Express server with GraphQL:

```javascript
// server.js
const express = require('express');
const { graphqlHTTP } = require('express-graphql');
const { buildSchema } = require('graphql');

// Define a GraphQL schema
const schema = buildSchema(`
  type Query {
    hello: String
    greet(name: String!): String
  }
`);

// Define the resolver functions
const root = {
  hello: () => {
    return 'Hello, world!';
  },
  greet: ({ name }) => {
    return `Hello, ${name}!`;
  },
};

const app = express();
const PORT = process.env.PORT || 4000;

// Set up the GraphQL endpoint
app.use('/graphql', graphqlHTTP({
  schema: schema,
  rootValue: root,
  graphiql: true, // Enable GraphiQL interface for testing
}));

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}/graphql`);
});
```

### Step 3: Start the Server

Run your server with the following command:

```bash
node server.js
```

You should see output indicating that the server is running:

```
Server is running on http://localhost:4000/graphql
```

### Step 4: Test Your GraphQL API

1. **Open Your Browser**

   Navigate to `http://localhost:4000/graphql`. You should see the GraphiQL interface, which allows you to test your GraphQL queries.

2. **Run a Query**

   In the left panel of GraphiQL, you can test your queries. Try the following queries:

   ```graphql
   {
     hello
   }
   ```

   This should return:

   ```json
   {
     "data": {
       "hello": "Hello, world!"
     }
   }
   ```

3. **Test the Greet Query**

   You can also test the `greet` query with a name:

   ```graphql
   {
     greet(name: "Alice")
   }
   ```

   This should return:

   ```json
   {
     "data": {
       "greet": "Hello, Alice!"
     }
   }
   ```

### Step 5: Expand Your GraphQL API

You can expand your GraphQL API by adding more types and queries. For example, let's add a type for `User` and a query to fetch user information.

1. **Update the Schema**

Modify your schema to include a `User` type:

```javascript
// Define a User type and update the Query type
const schema = buildSchema(`
  type User {
    id: ID
    name: String
    age: Int
  }

  type Query {
    hello: String
    greet(name: String!): String
    user(id: ID!): User
  }
`);
```

2. **Define Resolvers for User**

Now add a resolver function for the `user` query:

```javascript
const users = [
  { id: 1, name: 'Alice', age: 30 },
  { id: 2, name: 'Bob', age: 25 },
];

const root = {
  hello: () => {
    return 'Hello, world!';
  },
  greet: ({ name }) => {
    return `Hello, ${name}!`;
  },
  user: ({ id }) => {
    return users.find(user => user.id == id);
  },
};
```

3. **Test the User Query**

You can now test the `user` query in GraphiQL:

```graphql
{
  user(id: 1) {
    id
    name
    age
  }
}
```

This should return:

```json
{
  "data": {
    "user": {
      "id": "1",
      "name": "Alice",
      "age": 30
    }
  }
}
```

### Conclusion

You have successfully set up a basic GraphQL API using Express.js! This setup can be further expanded with additional features such as:

- **Mutations**: To allow creating, updating, or deleting data.
- **Middleware**: For authentication or logging.
- **Database Integration**: Using MongoDB, PostgreSQL, or another database to store your data.
- **Custom Directives**: For more advanced functionalities.

GraphQL provides a flexible and powerful way to interact with your API, making it an excellent choice for modern applications.