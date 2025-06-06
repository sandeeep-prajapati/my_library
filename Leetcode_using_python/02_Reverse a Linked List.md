### **Reversing a Singly Linked List**

---

### **Understanding the Problem**
You are given the head of a singly linked list. Your task is to reverse the linked list such that the head becomes the tail, and the tail becomes the head.

---

### **Approach**

#### **1. Iterative Approach**
This approach uses three pointers to reverse the linked list.

#### **Key Insight**
- As you traverse the list, reverse the direction of the `next` pointer for each node.
- Use three pointers:
  1. **`prev`**: Tracks the previous node.
  2. **`curr`**: Tracks the current node.
  3. **`next`**: Temporarily stores the next node before reversing the pointer.

#### **Algorithm**
1. Initialize:
   - `prev = None` (there is no previous node at the start).
   - `curr = head` (the head of the list is the current node).
2. Traverse the list:
   - Temporarily store the next node: `next = curr.next`.
   - Reverse the pointer: `curr.next = prev`.
   - Move the pointers forward:
     - `prev = curr`.
     - `curr = next`.
3. At the end, `prev` will point to the new head of the reversed list.

#### **Complexity**
- **Time**: \(O(n)\) — Every node is visited once.
- **Space**: \(O(1)\) — Only a few pointers are used.

---

### **Python Code (Iterative)**

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list(head):
    prev = None
    curr = head
    
    while curr:
        # Save the next node
        next = curr.next
        
        # Reverse the current node's pointer
        curr.next = prev
        
        # Move pointers one step forward
        prev = curr
        curr = next
    
    # Return the new head of the reversed list
    return prev
```

---

### **Example Walkthrough**

#### Input:  
`head = [1 -> 2 -> 3 -> 4 -> None]`

#### Execution:
1. **Initial state**:
   - `prev = None`
   - `curr = 1`
2. **First iteration**:
   - `next = 2`
   - Reverse: `curr.next = prev` → `1 -> None`
   - Move forward: `prev = 1`, `curr = 2`
3. **Second iteration**:
   - `next = 3`
   - Reverse: `curr.next = prev` → `2 -> 1 -> None`
   - Move forward: `prev = 2`, `curr = 3`
4. **Third iteration**:
   - `next = 4`
   - Reverse: `curr.next = prev` → `3 -> 2 -> 1 -> None`
   - Move forward: `prev = 3`, `curr = 4`
5. **Fourth iteration**:
   - `next = None`
   - Reverse: `curr.next = prev` → `4 -> 3 -> 2 -> 1 -> None`
   - Move forward: `prev = 4`, `curr = None`

#### Final state:
- `curr = None`
- `prev = 4` (new head of the reversed list).

#### Output:
`[4 -> 3 -> 2 -> 1 -> None]`

---

### **2. Recursive Approach**
Recursively reverse the list by processing one node at a time and flipping its pointer.

#### **Algorithm**
1. Base case:
   - If `head` is `None` or `head.next` is `None`, return `head` (end of list).
2. Recursive case:
   - Recursively reverse the rest of the list: `new_head = reverse_linked_list(head.next)`.
   - Flip the pointer of the next node: `head.next.next = head`.
   - Disconnect the current node: `head.next = None`.

#### **Complexity**
- **Time**: \(O(n)\) — Each node is visited once.
- **Space**: \(O(n)\) — Due to recursive function calls.

---

### **Python Code (Recursive)**

```python
def reverse_linked_list_recursive(head):
    # Base case: end of the list
    if not head or not head.next:
        return head
    
    # Reverse the rest of the list
    new_head = reverse_linked_list_recursive(head.next)
    
    # Reverse the current node's pointer
    head.next.next = head
    head.next = None
    
    return new_head
```

---

### **Tricks and Insights**
1. **Iterative vs Recursive**:
   - Use the iterative approach for better space efficiency.
   - Use recursion when you need simpler code, but beware of stack overflow for large lists.

2. **Debugging**:
   - Print the values of `prev`, `curr`, and `next` during iterations to understand the pointer changes.
   
3. **Visualization**:
   - Draw the linked list and arrows for `prev`, `curr`, and `next` to see how pointers are updated step by step.

---

### **Edge Cases**
1. **Empty List**: Input `head = None`.  
   - Output: `None`.
2. **Single Node**: Input `head = [1 -> None]`.  
   - Output: `[1 -> None]`.
3. **Multiple Nodes**: Input `head = [1 -> 2 -> 3]`.  
   - Output: `[3 -> 2 -> 1]`.
