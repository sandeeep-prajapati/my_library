Using Hugging Face Transformers with PyTorch for text classification tasks like sentiment analysis is straightforward and allows you to leverage pre-trained models, fine-tune them on your data, and make predictions with ease. Here’s a step-by-step guide:

### 1. Set Up Your Environment
First, make sure you have the necessary libraries installed:

```bash
pip install torch transformers datasets
```

### 2. Load a Pre-trained Transformer Model
For sentiment analysis, `distilbert-base-uncased` or `bert-base-uncased` are popular choices. These models are pre-trained on vast corpora and can be fine-tuned for classification.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels for binary classification
```

### 3. Preprocess the Text Data
Tokenize the input text and convert it to a format that PyTorch can process. For text classification, you typically need input IDs and attention masks.

```python
text = "The movie was fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

print(inputs["input_ids"])  # Encoded text
print(inputs["attention_mask"])  # Attention mask
```

### 4. Prepare the Dataset
To fine-tune the model, you'll need a labeled dataset. For this example, use the `datasets` library to load a pre-existing sentiment dataset like IMDb or create a custom dataset.

```python
from datasets import load_dataset

# Load IMDb sentiment dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]
```

### 5. Tokenize the Dataset
Batch process and tokenize the dataset using the tokenizer. Define a helper function to tokenize each data sample.

```python
def tokenize_data(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Apply the tokenizer to the train and test data
train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
```

### 6. Create Data Loaders
Create PyTorch `DataLoader` objects to load data in batches, which helps optimize the training process.

```python
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

### 7. Define the Training Loop
Use PyTorch to define a training loop that optimizes the model on the dataset. You’ll need an optimizer and a loss function.

```python
from transformers import AdamW
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 2
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_loss}")
```

### 8. Evaluate the Model
Once the model is trained, evaluate its accuracy on the test dataset.

```python
from torch.nn.functional import softmax

model.eval()
total_correct = 0
total_samples = 0

for batch in test_dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)
        
        # Calculate accuracy
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.2f}")
```

### 9. Inference
After training, you can use the model for inference on new texts:

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = "positive" if probs[0][1] > probs[0][0] else "negative"
    return sentiment

text = "This product is amazing!"
print(f"Sentiment: {predict_sentiment(text)}")
```

### Summary
- **Load** a pre-trained Hugging Face model and tokenizer.
- **Tokenize** and preprocess text data for model input.
- **Train** the model on labeled data using PyTorch.
- **Evaluate** accuracy on a test set.
- **Predict** sentiment on new text data.

This setup allows you to efficiently classify text using Hugging Face Transformers and PyTorch, making it ideal for sentiment analysis and similar NLP classification tasks.