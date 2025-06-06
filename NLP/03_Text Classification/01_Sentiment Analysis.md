### Sentiment Analysis

Sentiment analysis is the process of determining whether a piece of text expresses a positive, negative, or neutral opinion. It is widely used in applications like customer feedback analysis, product reviews, social media monitoring, and more.

#### **Steps to Perform Sentiment Analysis:**

1. **Data Collection**: Obtain text data that contains sentiment information (e.g., product reviews, tweets).
2. **Text Preprocessing**: Clean and prepare the text data (e.g., tokenization, stopword removal).
3. **Feature Extraction**: Convert the text into a numerical format for the machine learning model (e.g., Bag of Words, TF-IDF, Word2Vec, or BERT embeddings).
4. **Model Building**: Train a machine learning or deep learning model (e.g., Naive Bayes, Logistic Regression, RNN, or BERT) to predict sentiment labels.
5. **Evaluation**: Test the model on unseen data and evaluate its accuracy using metrics like precision, recall, F1-score.

---

### Example 1: **Sentiment Analysis Using NLTK (Naive Bayes Classifier)**

This approach uses the Naive Bayes Classifier from the Natural Language Toolkit (NLTK).

#### **Step 1: Import Libraries**

```python
import nltk
from nltk.corpus import movie_reviews
import random

# Download necessary datasets
nltk.download('movie_reviews')
nltk.download('punkt')
```

#### **Step 2: Load and Preprocess Data**
We will use the `movie_reviews` dataset from NLTK, which consists of movie reviews categorized into "pos" (positive) and "neg" (negative).

```python
# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Preprocessing: Convert the list of words into a feature dictionary
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# Choose the top 2000 most common words as features
word_features = list(all_words.keys())[:2000]

# Function to extract features from a document
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Create feature sets for all documents
featuresets = [(document_features(doc), category) for (doc, category) in documents]
```

#### **Step 3: Train the Naive Bayes Classifier**

```python
# Split the data into training and testing sets (80% train, 20% test)
train_set, test_set = featuresets[:1600], featuresets[1600:]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display the most informative features
classifier.show_most_informative_features(10)
```

---

### Example 2: **Sentiment Analysis Using PyTorch (LSTM)**

For a more advanced approach, we can use deep learning models like **LSTM (Long Short-Term Memory)** for sentiment classification.

#### **Step 1: Import Libraries**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets
```

#### **Step 2: Data Loading and Preprocessing**

We will use `torchtext` for text processing and loading datasets like **IMDb** for sentiment analysis.

```python
# Define fields for the text and labels
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

# Load the IMDb dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Build the vocabulary (use GloVe embeddings)
TEXT.build_vocab(train_data, max_size=25_000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Create iterators for the data
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), batch_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

#### **Step 3: Define the LSTM Model**

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))
```

#### **Step 4: Train the Model**

```python
# Model parameters
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1  # Binary classification (positive or negative)

# Initialize model, optimizer, and loss function
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Move model and criterion to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# Training function
def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_iterator, optimizer, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
```

#### **Step 5: Evaluate the Model**

```python
# Evaluation function
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluate on test data
test_loss = evaluate_model(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.4f}')
```

---

### 3. **Evaluation Metrics**

- **Accuracy**: Percentage of correctly predicted sentiment labels.
- **Precision**: True positive predictions as a proportion of all positive predictions.
- **Recall**: True positive predictions as a proportion of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

You can use these metrics to evaluate the performance of both the Naive Bayes and LSTM models.

---

### Summary

- **NLTK Approach (Naive Bayes)**: A classical method using tokenization, feature extraction, and a probabilistic model. It is fast and effective for small-scale datasets.
- **PyTorch Approach (LSTM)**: A deep learning-based model suitable for more complex datasets. It captures sequential dependencies in the text through LSTMs, making it powerful for sentiment analysis tasks.

Both approaches are effective depending on the complexity of your data and the use case.