### **LRU Cache Implementation**

---

### **Prompt**  
**Goal**: Implement an LRU (Least Recently Used) cache with the following operations:
- `get(key)`: Retrieve the value from the cache if the key exists, otherwise return `-1`.
- `put(key, value)`: Insert a key-value pair into the cache. If the cache exceeds its capacity, evict the least recently used key.

---

### **Trick**  
- **Combination of Dictionary and Doubly Linked List**:
  - **Dictionary**: Provides O(1) access to cache items by key.
  - **Doubly Linked List**: Keeps track of the access order, allowing O(1) removals and additions at both ends (head for most recently used and tail for least recently used).

---

### **Methodology**  
1. **Doubly Linked List**:
   - The list will be used to track the order of accesses. The most recently accessed item is moved to the front (head), and the least recently used item is at the end (tail).
   - Each node in the list will store a `key` and `value`.
   
2. **Operations**:
   - **`get(key)`**:
     - If the key exists in the cache, move the corresponding node to the front of the doubly linked list (marking it as recently used).
     - Return the value.
     - If the key does not exist, return `-1`.
   - **`put(key, value)`**:
     - If the key already exists, update the value and move the node to the front of the list.
     - If the key is new, add a new node to the front. If the cache exceeds its capacity, remove the node from the tail (the least recently used node).

3. **Eviction**:
   - If the cache exceeds its capacity, remove the least recently used item from the list (tail of the doubly linked list).

---

### **Python Code Implementation**:

```python
class LRUCache:
    class DllNode:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # HashMap to store key -> node mapping
        self.head = self.DllNode(0, 0)  # Dummy head node
        self.tail = self.DllNode(0, 0)  # Dummy tail node
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove_node(self, node: DllNode):
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_node_to_front(self, node: DllNode):
        """Add a node to the front of the doubly linked list."""
        next_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = next_node
        next_node.prev = node
    
    def get(self, key: int) -> int:
        """Return the value of the key if present, else return -1."""
        if key in self.cache:
            node = self.cache[key]
            # Move the accessed node to the front (most recently used)
            self._remove_node(node)
            self._add_node_to_front(node)
            return node.value
        return -1
    
    def put(self, key: int, value: int) -> None:
        """Insert or update the value of the key."""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            # Move the updated node to the front (most recently used)
            self._remove_node(node)
            self._add_node_to_front(node)
        else:
            # If the cache is at capacity, remove the least recently used item (tail)
            if len(self.cache) >= self.capacity:
                tail_node = self.tail.prev
                self._remove_node(tail_node)
                del self.cache[tail_node.key]
            # Add the new node to the front
            new_node = self.DllNode(key, value)
            self.cache[key] = new_node
            self._add_node_to_front(new_node)

# Example usage:
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))   # Output: 1
cache.put(3, 3)       # Evicts key 2
print(cache.get(2))   # Output: -1 (not found)
cache.put(4, 4)       # Evicts key 1
print(cache.get(1))   # Output: -1 (not found)
print(cache.get(3))   # Output: 3
print(cache.get(4))   # Output: 4
```

---

### **Explanation of the Code**:

1. **`DllNode` class**:
   - This is a helper class to represent each node in the doubly linked list. Each node has:
     - `key` and `value` attributes.
     - `prev` and `next` pointers to link nodes in the doubly linked list.

2. **LRUCache Class**:
   - **Initialization**:
     - We initialize the `capacity`, a `cache` dictionary for fast lookup, and two dummy nodes (`head` and `tail`) that represent the boundaries of the doubly linked list.
   - **Helper Methods**:
     - `_remove_node(node)`: Removes a node from the doubly linked list.
     - `_add_node_to_front(node)`: Adds a node to the front of the doubly linked list (making it the most recently used).
   - **`get(key)`**: If the key exists, retrieve the corresponding node, move it to the front (most recently used), and return its value. Otherwise, return `-1`.
   - **`put(key, value)`**: If the key exists, update its value and move it to the front. If the key doesn't exist:
     - If the cache is at capacity, remove the least recently used node from the tail and evict it from the dictionary.
     - Add the new node to the front.

---

### **Time Complexity**:
- **`get(key)`**: O(1), as it requires O(1) operations for both lookup and list reordering.
- **`put(key, value)`**: O(1), as it involves O(1) operations for both inserting/updating the node and list reordering. In case of eviction, the operation also takes O(1).

### **Space Complexity**:
- O(capacity), as the space used is proportional to the number of nodes stored in the cache, which is bounded by the cache's capacity.

---

### **Edge Cases**:
1. **Cache at Full Capacity**:
   - When the cache is full, the least recently used key is evicted when a new key is inserted.
   
2. **Accessing Evicted Keys**:
   - Once a key is evicted, attempting to `get(key)` will return `-1`.
   
3. **Cache with Minimum Capacity**:
   - The cache can handle a capacity of 1, in which case only one item can be stored. If another item is added, the old item is evicted.

---

### **Example Walkthrough**:

- **`put(1, 1)`**: Adds `1:1` to the cache.
- **`put(2, 2)`**: Adds `2:2` to the cache.
- **`get(1)`**: Returns `1`, and moves key `1` to the front (most recently used).
- **`put(3, 3)`**: Evicts `2` (least recently used) and adds `3:3` to the cache.
- **`get(2)`**: Returns `-1` because `2` was evicted.
- **`put(4, 4)`**: Evicts `1` (least recently used) and adds `4:4` to the cache.
- **`get(1)`**: Returns `-1` because `1` was evicted.
- **`get(3)`**: Returns `3`, as it's in the cache.
- **`get(4)`**: Returns `4`, as it's in the cache.

This implementation ensures O(1) time complexity for both `get` and `put` operations, achieving an efficient LRU cache.