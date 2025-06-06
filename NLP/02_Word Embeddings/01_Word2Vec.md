### Word2Vec Using NLTK and PyTorch

**Word2Vec** is a popular word embedding technique introduced by Google. It is used to represent words in a continuous vector space where similar words have similar vector representations. It captures semantic relationships between words, making it useful for tasks like NLP, machine learning, and deep learning applications.

---

### 1. **Word2Vec Overview**

Word2Vec typically comes in two model architectures:
- **Continuous Bag of Words (CBOW)**: Predicts the target word from the context words.
- **Skip-Gram**: Predicts the context words given a target word.

Both models use neural networks to learn word representations by minimizing the loss between the predicted word and the actual word.

---

### 2. **Implementing Word2Vec with NLTK and PyTorch**

We'll cover the process of creating a Word2Vec model using **NLTK** to preprocess the text data and **PyTorch** to build the neural network.

#### **Step 1: Install the required libraries**
You need the following libraries to run this example:
```bash
pip install nltk torch
```

#### **Step 2: Preprocessing with NLTK**

We’ll use NLTK to tokenize and preprocess the text data. Here's how to download the necessary resources and tokenize sentences.

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Example corpus
text = """Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space."""
sentences = sent_tokenize(text)

# Tokenize words and remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_sentences = []
for sentence in sentences:
    words = word_tokenize(sentence.lower())  # Lowercasing for uniformity
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    tokenized_sentences.append(filtered_words)

# Flatten tokenized_sentences into a single list of words
vocab = Counter([word for sentence in tokenized_sentences for word in sentence])

# Display tokenized words
print("Tokenized Sentences:", tokenized_sentences)
print("Vocabulary:", vocab)
```

#### **Step 3: Create Skip-Gram Dataset**

Now, we need to create a dataset for the **Skip-Gram** model, where the input is the target word and the output is the context word. We'll generate word pairs from the corpus for training.

```python
import torch
import random

# Create a word-to-index mapping and vice versa
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Hyperparameters
context_window = 2  # Number of words before and after the target word

# Generate Skip-Gram word pairs
def generate_skipgram_data(tokenized_sentences, window_size):
    pairs = []
    for sentence in tokenized_sentences:
        for i, target_word in enumerate(sentence):
            context = sentence[max(0, i - window_size): i] + sentence[i+1: i + window_size + 1]
            for context_word in context:
                pairs.append((target_word, context_word))
    return pairs

skipgram_data = generate_skipgram_data(tokenized_sentences, context_window)
print("Skip-Gram Data Example:", skipgram_data[:5])

# Convert words to indexes for PyTorch
def word_to_idx_pairs(skipgram_data):
    return [(word_to_idx[target], word_to_idx[context]) for target, context in skipgram_data]

skipgram_data_idx = word_to_idx_pairs(skipgram_data)
```

#### **Step 4: Building the Word2Vec Model in PyTorch**

We'll create a simple **Skip-Gram Word2Vec** model using PyTorch. This involves two linear layers — the first to project the input word to a hidden vector and the second to output a vector in the vocabulary size.

```python
import torch.nn as nn
import torch.optim as optim

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, target_word):
        # Get the embedding of the target word
        embedding = self.embedding(target_word)
        output = self.output_layer(embedding)
        return output

# Hyperparameters
embedding_dim = 100
vocab_size = len(vocab)

# Instantiate the model
model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, data, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx in data:
            target_tensor = torch.tensor([target_idx], dtype=torch.long)
            context_tensor = torch.tensor([context_idx], dtype=torch.long)
            
            # Forward pass
            output = model(target_tensor)
            loss = criterion(output, context_tensor)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch: {epoch + 1}, Loss: {total_loss/len(data)}')

# Train the Word2Vec model
train_model(model, skipgram_data_idx, epochs=10)
```

#### **Step 5: Extracting Word Embeddings**

After training the model, you can extract the word embeddings by accessing the embedding layer.

```python
# Get the embedding of a word
def get_word_embedding(word):
    word_idx = torch.tensor([word_to_idx[word]], dtype=torch.long)
    return model.embedding(word_idx).detach().numpy()

# Example: Get the embedding for 'word'
embedding = get_word_embedding('word')
print("Embedding for 'word':", embedding)
```

---

### 3. **Summary**

1. **Data Preprocessing**:
   - We tokenized and preprocessed text using **NLTK** by removing stopwords and creating a vocabulary.
   
2. **Skip-Gram Data Generation**:
   - We created a Skip-Gram dataset with word pairs of target and context words.
   
3. **Word2Vec Model with PyTorch**:
   - Built a simple neural network model using PyTorch to learn word embeddings with a Skip-Gram approach.
   
4. **Training**:
   - The model was trained to minimize the loss between the predicted context words and actual context words.

5. **Word Embedding Extraction**:
   - Extracted word embeddings for specific words from the trained model.

You now have a basic Word2Vec model implemented using NLTK and PyTorch!