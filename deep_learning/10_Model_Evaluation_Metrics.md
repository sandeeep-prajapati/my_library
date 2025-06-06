# 10. **Model Evaluation Metrics**

## Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Using These Metrics for Model Evaluation

Evaluating the performance of a machine learning model is crucial for understanding its effectiveness and reliability. Different metrics provide insights into different aspects of a model's performance. Below, we’ll discuss key evaluation metrics used in classification tasks and how to implement them in PyTorch.

### 10.1 **Accuracy**

**Definition**: Accuracy is the ratio of correctly predicted instances to the total instances in the dataset.

**Formula**:
\[
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
\]

**Usage**: Accuracy is a straightforward metric but may not be sufficient for imbalanced datasets.

**Implementation in PyTorch**:
```python
def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
```

### 10.2 **Precision**

**Definition**: Precision, also known as Positive Predictive Value, measures the proportion of true positive predictions out of all positive predictions made by the model.

**Formula**:
\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

**Usage**: Precision is important when the cost of false positives is high, such as in medical diagnoses.

**Implementation in PyTorch**:
```python
def calculate_precision(predictions, labels):
    TP = ((predictions == 1) & (labels == 1)).sum().item()
    FP = ((predictions == 1) & (labels == 0)).sum().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision
```

### 10.3 **Recall**

**Definition**: Recall, or Sensitivity, measures the proportion of true positive predictions out of all actual positive instances.

**Formula**:
\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

**Usage**: Recall is crucial when the cost of false negatives is high, such as in fraud detection.

**Implementation in PyTorch**:
```python
def calculate_recall(predictions, labels):
    TP = ((predictions == 1) & (labels == 1)).sum().item()
    FN = ((predictions == 0) & (labels == 1)).sum().item()
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall
```

### 10.4 **F1-Score**

**Definition**: The F1-Score is the harmonic mean of precision and recall, providing a balance between the two metrics.

**Formula**:
\[
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**Usage**: The F1-Score is particularly useful for imbalanced classes, giving a better measure of the incorrectly classified cases.

**Implementation in PyTorch**:
```python
def calculate_f1_score(predictions, labels):
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score
```

### 10.5 **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**

**Definition**: ROC-AUC is a performance measurement for classification problems at various threshold settings. The ROC curve is a graphical plot of the true positive rate (Recall) against the false positive rate.

**Usage**: AUC represents the degree or measure of separability, indicating how well the model can distinguish between classes. A value of 1 indicates perfect prediction, while 0.5 indicates no discriminative power.

**Implementation in PyTorch**:
```python
from sklearn.metrics import roc_auc_score

def calculate_roc_auc(predictions, labels):
    roc_auc = roc_auc_score(labels.cpu(), predictions.cpu())
    return roc_auc
```

### 10.6 **Summary of Metrics**

| Metric             | Formula                                          | Use Case                              |
|--------------------|--------------------------------------------------|---------------------------------------|
| Accuracy           | \(\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}\) | General performance measure           |
| Precision          | \(\frac{\text{TP}}{\text{TP} + \text{FP}}\)    | Cost of false positives is high       |
| Recall             | \(\frac{\text{TP}}{\text{TP} + \text{FN}}\)    | Cost of false negatives is high       |
| F1-Score           | \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\) | Balance between precision and recall   |
| ROC-AUC            | Area under the ROC curve                        | Measure of separability               |

### 10.7 **Best Practices for Model Evaluation**

- **Multiple Metrics**: Always evaluate using multiple metrics to get a comprehensive understanding of model performance, especially in imbalanced datasets.
- **Confusion Matrix**: Use a confusion matrix for detailed insights into true positives, true negatives, false positives, and false negatives.
- **Cross-Validation**: Consider using k-fold cross-validation to ensure the model's robustness and to mitigate overfitting.

---

This section provides an overview of important model evaluation metrics, their definitions, formulas, usage scenarios, and practical implementations in PyTorch. Understanding these metrics is essential for assessing model performance effectively.