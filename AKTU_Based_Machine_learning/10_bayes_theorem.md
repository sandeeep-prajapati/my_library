### **Theory: Bayes' Theorem**

---

#### **Bayes' Theorem:**

Bayes' theorem provides a way to update the probability of a hypothesis based on new evidence. It is a fundamental concept in statistics and probability theory, particularly useful in machine learning, especially for classification problems.

The theorem is expressed as:

\[
P(H|E) = \frac{P(E|H)P(H)}{P(E)}
\]

Where:
- \(P(H|E)\) is the **posterior probability**, the probability of the hypothesis \(H\) being true given the evidence \(E\).
- \(P(E|H)\) is the **likelihood**, the probability of observing the evidence \(E\) given that the hypothesis \(H\) is true.
- \(P(H)\) is the **prior probability**, the initial probability of the hypothesis \(H\) before seeing the evidence.
- \(P(E)\) is the **marginal likelihood** or the total probability of observing the evidence.

---

#### **Explanation:**

- **Prior Probability** \(P(H)\): This represents our belief about the hypothesis before considering the current evidence. It’s the probability that the hypothesis is true, independent of the current data.
  
- **Likelihood** \(P(E|H)\): This is the probability of observing the evidence given the hypothesis is true. It reflects how likely the data is assuming the hypothesis holds.

- **Posterior Probability** \(P(H|E)\): This is the updated probability of the hypothesis after incorporating the evidence. It is what we are trying to calculate using Bayes' theorem.

- **Marginal Likelihood** \(P(E)\): This is the total probability of the evidence under all possible hypotheses. It serves as a normalizing constant to ensure that the posterior probabilities sum to one.

---

#### **Example:**

Suppose you have a classification problem with two classes: \(H_1\) (Class 1) and \(H_2\) (Class 2). You want to predict the probability of \(H_1\) (a certain class) given the evidence \(E\) (the feature value).

- Prior Probability: \(P(H_1)\) = 0.6 (60% of the data points belong to class 1)
- Likelihood: \(P(E|H_1)\) = 0.8 (80% probability of observing feature \(E\) if the data point is from class 1)
- Marginal Likelihood: \(P(E)\) = 0.7 (the total probability of observing \(E\))

Using Bayes’ theorem, the posterior probability \(P(H_1|E)\) is calculated as:

\[
P(H_1|E) = \frac{P(E|H_1) \cdot P(H_1)}{P(E)} = \frac{0.8 \cdot 0.6}{0.7} = 0.6857
\]

So, the probability of the data point belonging to class 1 after observing feature \(E\) is 68.57%.

---

### **Prompt: Calculate Posterior Probabilities Using PyTorch for a Simple Classification Problem**

Here is a PyTorch implementation of Bayes' theorem to calculate the posterior probability for a simple classification problem where we have two classes, \(H_1\) and \(H_2\).

```python
import torch

# Define the prior probabilities and likelihoods
P_H1 = torch.tensor(0.6)  # Prior probability of class 1
P_H2 = torch.tensor(0.4)  # Prior probability of class 2

P_E_given_H1 = torch.tensor(0.8)  # Likelihood of evidence given class 1
P_E_given_H2 = torch.tensor(0.3)  # Likelihood of evidence given class 2

P_E = torch.tensor(0.7)  # Marginal likelihood (probability of evidence)

# Using Bayes' theorem to calculate the posterior probabilities
P_H1_given_E = (P_E_given_H1 * P_H1) / P_E
P_H2_given_E = (P_E_given_H2 * P_H2) / P_E

# Display the results
print(f"Posterior probability of H1 given E: {P_H1_given_E.item():.4f}")
print(f"Posterior probability of H2 given E: {P_H2_given_E.item():.4f}")
```

---

### **Explanation of the Code:**

1. **Prior Probabilities:**  
   We define the prior probabilities for each class: \(P(H_1) = 0.6\) and \(P(H_2) = 0.4\), meaning that 60% of the data points are in class 1 and 40% in class 2.

2. **Likelihoods:**  
   The likelihoods \(P(E|H_1) = 0.8\) and \(P(E|H_2) = 0.3\) indicate the probability of observing the evidence \(E\) if the class is \(H_1\) or \(H_2\).

3. **Marginal Likelihood:**  
   We set \(P(E) = 0.7\), the total probability of observing \(E\), which is computed by summing the likelihoods weighted by the prior probabilities.

4. **Posterior Probabilities:**  
   Using Bayes' theorem, we calculate the posterior probabilities for each class given the evidence:
   \[
   P(H_1|E) = \frac{P(E|H_1) \cdot P(H_1)}{P(E)}
   \]
   and similarly for \(P(H_2|E)\).

5. **Results:**  
   The posterior probabilities of the two classes are printed out.

---

### **Key Takeaways:**

- **Bayes' theorem** is a powerful statistical tool that helps update the probability of a hypothesis based on new evidence.
- It is widely used in classification tasks, particularly in models like **Naïve Bayes** classifiers.
- In this example, we demonstrated how to calculate posterior probabilities using Bayes' theorem with PyTorch for a simple classification problem.
