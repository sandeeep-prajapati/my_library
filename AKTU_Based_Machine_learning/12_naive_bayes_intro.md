### **Theory: Naïve Bayes Classifier**

#### **Naïve Bayes Classifier:**

The **Naïve Bayes** classifier is a probabilistic classifier based on Bayes' Theorem, assuming that the features are conditionally independent given the class label. Despite its simplicity, it often performs surprisingly well, especially in text classification tasks.

Bayes' Theorem is expressed as:

\[
P(C_k | X) = \frac{P(X | C_k) P(C_k)}{P(X)}
\]

Where:
- \( P(C_k | X) \) is the posterior probability of class \( C_k \) given the feature vector \( X \).
- \( P(X | C_k) \) is the likelihood of the features \( X \) given the class \( C_k \).
- \( P(C_k) \) is the prior probability of class \( C_k \).
- \( P(X) \) is the evidence or the marginal likelihood of \( X \).

The **Naïve Bayes** classifier computes the class \( C_k \) that maximizes the posterior probability:

\[
C_{\text{optimal}} = \arg\max_k P(C_k | X)
\]

Since it assumes **conditional independence** of features given the class, the likelihood \( P(X | C_k) \) is computed as the product of individual feature likelihoods:

\[
P(X | C_k) = \prod_{i=1}^{n} P(x_i | C_k)
\]

Where \( x_i \) is the \( i \)-th feature in the vector \( X \) and \( n \) is the number of features.

The **Naïve Bayes** classifier can be applied in two common variants:
1. **Gaussian Naïve Bayes:** For continuous features, assuming that the features follow a Gaussian (normal) distribution.
2. **Multinomial Naïve Bayes:** For discrete features (typically used in text classification), where the features represent word frequencies or counts.

---

#### **Advantages of Naïve Bayes:**
1. **Simple and Fast**: Naïve Bayes is computationally efficient and works well for high-dimensional data.
2. **Easy to Implement**: It is easy to understand and implement, especially in text classification tasks like spam filtering and sentiment analysis.
3. **Good Performance with Small Datasets**: Even with limited training data, Naïve Bayes can yield good results, particularly when features are conditionally independent.
4. **Robust to Irrelevant Features**: The model can still perform well even when some features are irrelevant.

---

#### **Limitations of Naïve Bayes:**
1. **Conditional Independence Assumption**: The biggest limitation is the assumption that features are conditionally independent. This assumption often does not hold in real-world data (e.g., in text classification, where certain words are highly correlated).
2. **Poor Performance with Highly Correlated Features**: If the features are highly correlated (e.g., in images or certain types of text), Naïve Bayes may perform poorly compared to other models like decision trees or neural networks.
3. **Zero Probability Problem**: If any feature has a zero probability for a given class, it can make the entire likelihood zero. This issue is mitigated using **Laplace smoothing**.
4. **Limited to Classification**: Naïve Bayes is typically used only for classification tasks, and it cannot directly be used for regression.

---

### **Prompt: Implement Naïve Bayes in PyTorch to Categorize News Articles**

Below is an implementation of **Naïve Bayes** using **PyTorch** to classify news articles into different categories (e.g., sports, politics, technology).

We'll use the **Multinomial Naïve Bayes** approach, where the features are the word frequencies in the articles.

#### **Step-by-step Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample news articles
texts = [
    "Stock market hits new highs with rising tech stocks", 
    "The government passed a new bill on healthcare reform", 
    "Sports teams preparing for the upcoming championship",
    "New advancements in AI technology revolutionize industries",
    "Local elections coming up next week with hotly contested races",
    "Tech company releases groundbreaking smartphone",
    "Soccer players train for international competition"
]

# Corresponding labels for the articles (0: Politics, 1: Technology, 2: Sports)
labels = [0, 1, 2, 1, 0, 1, 2]

# Convert text data to bag-of-words features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Naive Bayes Classifier Model using PyTorch
class NaiveBayesClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NaiveBayesClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Log-prior probabilities for each class
        self.log_priors = nn.Parameter(torch.zeros(num_classes))
        # Log-likelihood for each feature in each class
        self.log_likelihoods = nn.Parameter(torch.zeros(num_classes, input_dim))
    
    def forward(self, X):
        # Compute the log-probabilities for each class
        log_probs = self.log_priors + X @ self.log_likelihoods.T
        return log_probs  # Return the raw log probabilities

# Initialize model, loss function, and optimizer
model = NaiveBayesClassifier(X_train.shape[1], len(set(labels)))  # Number of features and classes
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    y_pred = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_train_tensor)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    predicted = torch.argmax(y_pred_test, dim=1)  # Get class with highest probability
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f'Accuracy on test data: {accuracy * 100:.2f}%')
```

---

### **Explanation of the Code:**

1. **Data Preprocessing:**
   - We use the `CountVectorizer` from scikit-learn to convert the raw text data (news articles) into a **bag-of-words** representation. This results in a sparse matrix of word counts for each article.
   - The labels represent the categories: Politics (0), Technology (1), and Sports (2).
   
2. **Naive Bayes Model:**
   - The model consists of two components:
     - **Log-priors**: The prior probability of each class (category).
     - **Log-likelihoods**: The likelihood of each feature (word) occurring in each class.
   - The forward pass calculates the **log-probability** for each class and selects the class with the highest log-probability.

3. **Training the Model:**
   - We train the model using **Stochastic Gradient Descent (SGD)** for 100 epochs, minimizing the **cross-entropy loss**.
   - The parameters (`log_priors` and `log_likelihoods`) are updated using backpropagation.

4. **Evaluation:**
   - After training, we evaluate the model on the test set by computing the accuracy, i.e., how many test articles are correctly classified into their respective categories.

---

### **Key Takeaways:**

- The **Naïve Bayes** classifier is a simple yet effective model for text classification tasks, especially in cases where the features (words) are conditionally independent.
- Despite its limitations, such as the independence assumption, Naïve Bayes can perform well with text data, where features (words) are often treated as independent for simplicity.
- The **Multinomial Naïve Bayes** variant is particularly suited for text classification problems, where features are word counts or frequencies.
