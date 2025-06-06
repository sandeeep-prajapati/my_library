Fine-tuning a GPT-based model for text generation tasks using PyTorch and the Hugging Face Transformers library involves several steps. Here’s a step-by-step guide to help you through the process:

### 1. Set Up Your Environment
First, make sure you have the necessary libraries installed:

```bash
pip install torch transformers datasets
```

### 2. Load Your Custom Dataset
Prepare your dataset for the text generation task. This dataset should ideally be a text file where each line represents a separate example. For this example, let’s create a simple text file.

```python
# Sample text data
text_data = """Once upon a time in a faraway land, there was a brave knight.
The knight fought valiantly to protect his kingdom.
He faced many challenges along the way.
In the end, good prevailed over evil."""

# Save it to a text file
with open("custom_text_data.txt", "w") as f:
    f.write(text_data)
```

### 3. Load the Pre-trained GPT Model and Tokenizer
Choose a pre-trained GPT model. Here, we will use `gpt2`.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 4. Prepare the Dataset for Training
Load and preprocess the dataset for training. The dataset will need to be tokenized.

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("text", data_files="custom_text_data.txt")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Keep only input_ids for the model
tokenized_datasets.set_format(type="torch", columns=["input_ids"])
```

### 5. Create Data Loaders
Use PyTorch’s `DataLoader` to create iterable datasets for training.

```python
from torch.utils.data import DataLoader

batch_size = 2  # Adjust based on your GPU memory
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
```

### 6. Define the Training Loop
Set up the training loop, specifying the optimizer and loss function.

```python
from transformers import AdamW
import torch

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)

        # Shift input_ids for the language modeling task
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_loss:.4f}")
```

### 7. Save the Fine-tuned Model
After training, save the fine-tuned model for later use.

```python
model.save_pretrained("fine_tuned_gpt_model")
tokenizer.save_pretrained("fine_tuned_gpt_model")
```

### 8. Generate Text with the Fine-tuned Model
You can now use the fine-tuned model to generate text based on a prompt.

```python
def generate_text(prompt, max_length=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example text generation
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
```

### Summary
1. **Prepare Your Dataset**: Create a text file containing the data for your text generation task.
2. **Load the Pre-trained Model**: Select a GPT model and load its tokenizer and configuration.
3. **Tokenize Your Data**: Use the tokenizer to preprocess the text into input IDs.
4. **Train the Model**: Define a training loop to fine-tune the model on your dataset.
5. **Save the Model**: Store the fine-tuned model for future use.
6. **Generate Text**: Use the model to generate text based on a given prompt.

This approach allows you to leverage the capabilities of GPT models for your specific text generation tasks while adapting them to your unique dataset.