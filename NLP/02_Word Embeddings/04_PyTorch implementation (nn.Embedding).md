### PyTorch Implementation of Word2Vec Using `nn.Embedding`

The `nn.Embedding` layer in PyTorch is used to represent words or categorical variables as dense vectors. This layer is perfect for implementing the **Word2Vec** model, which transforms words into continuous vector space representations.

In this example, we will implement a Skip-Gram Word2Vec model using PyTorch's `nn.Embedding` layer to learn the word embeddings.

---

### 1. **Word2Vec with PyTorch: Skip-Gram Model**

#### **Step 1: Import Libraries**
We'll use PyTorch for building the neural network and NLTK for text preprocessing.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
```

#### **Step 2: Preprocessing the Text Data**
We'll tokenize the text and prepare the word pairs required for the Skip-Gram model. The dataset is based on word-context pairs.

```python
# Sample text
text = "Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space."

# Tokenize the sentence into words
tokenized_words = word_tokenize(text.lower())

# Count the occurrences of each word
vocab = Counter(tokenized_words)

# Word to index and index to word dictionaries
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Generate Skip-Gram word pairs (target, context)
def generate_skipgram_data(tokenized_words, window_size=2):
    pairs = []
    for i, target_word in enumerate(tokenized_words):
        context_words = tokenized_words[max(0, i-window_size): i] + tokenized_words[i+1: i+1+window_size]
        for context_word in context_words:
            pairs.append((target_word, context_word))
    return pairs

skipgram_data = generate_skipgram_data(tokenized_words)
```

#### **Step 3: Convert Words to Indexes**
Convert the word pairs into index pairs for use in the neural network.

```python
# Convert word pairs to index pairs
def word_to_idx_pairs(skipgram_data):
    return [(word_to_idx[target], word_to_idx[context]) for target, context in skipgram_data]

skipgram_data_idx = word_to_idx_pairs(skipgram_data)
```

#### **Step 4: Define the Word2Vec Model Using `nn.Embedding`**
We'll create a neural network model with an embedding layer to learn the word vectors. The model consists of two main layers:
1. `nn.Embedding`: This maps the input words to dense vector representations (word embeddings).
2. `nn.Linear`: This maps the embedding to the context words.

```python
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word):
        # Get the embedding of the target word
        embedding = self.embedding(target_word)
        # Output a score for each word in the vocabulary (predict context)
        output = self.output_layer(embedding)
        return output
```

#### **Step 5: Training the Word2Vec Model**

We will now train the model using the **negative log likelihood loss** (CrossEntropyLoss) and the **Adam optimizer**.

```python
# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 100  # Dimension of the word embedding vectors
learning_rate = 0.001
epochs = 10

# Instantiate the model, loss function, and optimizer
model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert the skipgram data to tensors
def data_to_tensor(skipgram_data_idx):
    return [(torch.tensor([target], dtype=torch.long), torch.tensor([context], dtype=torch.long))
            for target, context in skipgram_data_idx]

skipgram_tensor_data = data_to_tensor(skipgram_data_idx)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for target_word, context_word in skipgram_tensor_data:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(target_word)
        
        # Compute the loss
        loss = criterion(output, context_word)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(skipgram_tensor_data)}')
```

#### **Step 6: Extracting Word Embeddings**

After training the model, we can extract the word embeddings from the embedding layer for any word in the vocabulary.

```python
# Function to get the word embedding for a given word
def get_word_embedding(word):
    word_idx = torch.tensor([word_to_idx[word]], dtype=torch.long)
    return model.embedding(word_idx).detach().numpy()

# Example: Get the embedding for the word 'word'
embedding = get_word_embedding('word')
print("Embedding for 'word':", embedding)
```

---

### 2. **Explanation**

- **Embedding Layer (`nn.Embedding`)**: The embedding layer takes the index of a word as input and outputs a dense vector (embedding) for that word.
- **Output Layer**: A fully connected layer that outputs a score for each word in the vocabulary (used for context prediction in Skip-Gram).
- **Loss Function**: We use CrossEntropyLoss to minimize the difference between the predicted context word and the actual context word.

---

### 3. **Summary**

1. **Preprocessing**:
   - Tokenized text into words and created word-to-index mappings.
   - Generated Skip-Gram word pairs from the tokenized text.
   
2. **Model**:
   - Created a neural network with an `nn.Embedding` layer to represent words as vectors.
   - Used an output layer to predict context words for a given target word.

3. **Training**:
   - Trained the model using a CrossEntropyLoss function and optimized it with the Adam optimizer.

4. **Embedding Extraction**:
   - Extracted the learned word embeddings using the trained embedding layer.

This implementation uses PyTorch's `nn.Embedding` to efficiently represent words in a dense, continuous vector space, suitable for downstream NLP tasks!