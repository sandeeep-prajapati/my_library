Creating references between documents in MongoDB allows you to establish relationships between different collections, enabling a more normalized data structure. Here's how to create references, along with the pros and cons compared to embedding.

### Creating References in MongoDB

To create references, you typically store the ObjectId of one document in another document. For example, consider two collections: `users` and `posts`. Each post references a user who created it.

#### Example:

1. **User Document**:
   ```json
   {
     "_id": ObjectId("60c72b2f9b1d8c4f8f3a62d7"),
     "name": "Alice"
   }
   ```

2. **Post Document**:
   ```json
   {
     "_id": ObjectId("60c72b2f9b1d8c4f8f3a62d8"),
     "title": "My First Post",
     "content": "This is my first post.",
     "author": ObjectId("60c72b2f9b1d8c4f8f3a62d7")  // Reference to the user
   }
   ```

### Querying with References

To retrieve data that involves references, you often perform two queries: one to fetch the main document and another to fetch the referenced document.

#### Example Query in Mongoose:

```javascript
const Post = mongoose.model('Post', postSchema);
const User = mongoose.model('User', userSchema);

// Fetch a post and the author details
Post.findById(postId)
  .populate('author')  // Populates the author field with user details
  .exec((err, post) => {
    if (err) throw err;
    console.log(post);
  });
```

### Pros and Cons of Using References

#### Pros:

1. **Normalization**:
   - References help maintain a normalized data structure, which can reduce redundancy and help in maintaining data integrity.

2. **Flexibility**:
   - You can independently manage the documents. If an author’s details change, you only need to update the user document without needing to rewrite all related posts.

3. **Scalability**:
   - When dealing with large datasets, references can make it easier to manage the size of individual documents, as opposed to embedding large arrays.

4. **Complex Queries**:
   - With references, you can perform complex queries that might involve multiple collections more flexibly, as you are not limited to the structure of a single document.

5. **Independent Lifecycle**:
   - If the related data has a different lifecycle than the parent document, references allow for easier management and updates.

#### Cons:

1. **Performance Overhead**:
   - Fetching related data requires multiple queries, which can introduce latency and impact performance, especially if not optimized properly (e.g., using indexing).

2. **Data Consistency**:
   - Maintaining consistency across collections can be more challenging. If a referenced document is deleted or modified, you must ensure that any references in other documents are appropriately handled.

3. **Increased Complexity**:
   - Managing references can add complexity to your application code, as you need to handle multiple queries and ensure that the relationships are maintained.

4. **Joins**:
   - While MongoDB does support some join-like operations with the `$lookup` aggregation stage, they are generally less efficient than traditional SQL joins, especially with large datasets.

### Conclusion

Using references in MongoDB is suitable when you need to maintain a normalized database structure with independent data lifecycles. They provide flexibility and scalability at the cost of performance and complexity. On the other hand, embedding documents can improve performance and simplify retrieval for closely related data, but may lead to data duplication and larger document sizes. The choice between using references and embedding should be based on the specific use case, data access patterns, and the nature of the relationships between your data entities.