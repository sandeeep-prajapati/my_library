
---

## 🔗 **05\_learning\_about\_simplechain\_sequentialchain\_llmchain.md**

> 📌 *Topic: Learn about `SimpleChain`, `SequentialChain`, and `LLMChain` with code examples*

---

## 🧠 What Are Chains in LangChain?

A **chain** in LangChain is a modular component that **connects multiple steps together** — usually involving LLMs, prompts, and memory — to perform a task.

LangChain provides various chain types to suit different complexity levels.

---

## 1. 📦 **LLMChain (Most Common)**

### ✅ What is it?

`LLMChain` is the most basic chain. It takes:

* A `PromptTemplate`
* An `LLM` model (like OpenAI)
* An optional memory

Then runs the LLM with the prompt and variables.

---

### 🧪 Code Example:

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# 1. Create prompt
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# 2. Create LLM instance
llm = OpenAI(temperature=0.7)

# 3. Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Run the chain
print(chain.run("electric bikes"))
```

---

## 2. 🔀 **SimpleSequentialChain**

### ✅ What is it?

`SimpleSequentialChain` runs a **sequence of LLMChains**, **passing the output of one directly as input to the next**.

It’s useful when the output of one step is raw input for the next.

---

### 🧪 Code Example:

```python
from langchain.chains import SimpleSequentialChain

# First chain: summarize
prompt1 = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n{text}"
)
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Second chain: translate
prompt2 = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text to French:\n{text}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine chains
simple_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

input_text = "LangChain is a powerful framework for building LLM-powered apps."
result = simple_chain.run(input_text)
print(result)
```

---

## 3. 🔁 **SequentialChain**

### ✅ What is it?

`SequentialChain` is more powerful than `SimpleSequentialChain`.
It **allows named inputs/outputs**, so you can **reuse and combine values** across multiple steps.

Perfect for when:

* You have multiple inputs
* You want to pass outputs selectively
* Each chain uses different variable names

---

### 🧪 Code Example:

```python
from langchain.chains import SequentialChain

# First chain: generate product description
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="Write a short description for a product called {product}."
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="description")

# Second chain: write a tweet based on description
prompt2 = PromptTemplate(
    input_variables=["description"],
    template="Write a tweet to promote this product:\n{description}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="tweet")

# Combine into a SequentialChain
seq_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["product"],
    output_variables=["description", "tweet"],
    verbose=True
)

result = seq_chain.run({"product": "Smart Coffee Mug"})
print(result)
```

---

## 🔄 Summary of Differences

| Feature              | `LLMChain`          | `SimpleSequentialChain` | `SequentialChain`      |
| -------------------- | ------------------- | ----------------------- | ---------------------- |
| Input Type           | Single step         | Single output to input  | Named inputs/outputs   |
| Step Count           | 1                   | Multiple, direct pipe   | Multiple, flexible map |
| Custom Variable Flow | ❌                   | ❌                       | ✅                      |
| Best Use Case        | Simple prompt + LLM | Basic pipelines         | Complex pipelines      |

---

## ✅ Final Tips

* Start with `LLMChain` for isolated tasks
* Use `SimpleSequentialChain` for quick-and-dirty multi-step flows
* Use `SequentialChain` for **real-world apps** needing control and variable mapping

---
