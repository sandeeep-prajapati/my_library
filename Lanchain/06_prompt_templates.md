
---

## 🧾 **07\_mastering\_prompttemplate\_dynamic\_prompt\_generation.md**

> 📌 *Topic: Master `PromptTemplate`: dynamic prompt generation and variable injection*

---

## 🧠 What Is a PromptTemplate?

A `PromptTemplate` in LangChain allows you to **create flexible, parameterized prompts** that can dynamically insert user-defined values at runtime.

Think of it as the **"HTML template" of prompts**, where you plug in different content to get dynamic output.

---

## 📦 Why Use `PromptTemplate`?

| Feature          | Benefit                                          |
| ---------------- | ------------------------------------------------ |
| 🔄 Reusability   | Use the same prompt format with different inputs |
| 🧩 Modularity    | Clean separation of prompt logic and data        |
| ⚙️ Customization | Easy to version and tune prompts                 |
| 📊 Consistency   | Enforces structure across prompts                |

---

## 🧪 Basic Usage

### ✅ Simple Example:

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Format the prompt with data
formatted_prompt = prompt.format(product="eco-friendly bags")
print(formatted_prompt)
```

> 💡 Output:
> *"What is a good name for a company that makes eco-friendly bags?"*

---

## ⚙️ Input Variables

You must explicitly define **`input_variables`** — a list of all placeholders used in the template.

```python
input_variables=["product", "audience"]
template = "Suggest a name for a {product} targeting {audience}."
```

### 🚫 Error Example

If your template includes a placeholder not declared in `input_variables`, you’ll get a runtime error.

---

## 🧱 Advanced Example: Multi-Variable Prompt

```python
prompt = PromptTemplate(
    input_variables=["topic", "tone"],
    template="""
Write a {tone} paragraph explaining the importance of {topic}.
"""
)

formatted = prompt.format(topic="data privacy", tone="professional")
print(formatted)
```

---

## 🧠 Few-Shot Prompting with PromptTemplate

You can integrate examples into the prompt structure manually or use **`FewShotPromptTemplate`** (an advanced tool).

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "sun", "output": "A bright celestial body."},
    {"input": "moon", "output": "Earth’s natural satellite."},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: What is {input}?\nA: {output}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer the following questions:",
    suffix="Q: What is {input}?\nA:",
    input_variables=["input"]
)

print(prompt.format(input="stars"))
```

---

## 📚 PromptTemplate with LLMChain

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["country"],
    template="What are the top 3 tourist attractions in {country}?"
)

llm = OpenAI(temperature=0.5)
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("India")
print(response)
```

---

## 🧰 PromptTemplate with `format_messages` (for Chat Models)

LangChain supports **chat models** with `ChatPromptTemplate`:

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Translate this sentence to French: {sentence}"
)

messages = prompt.format_messages(sentence="I love open-source AI.")
print(messages)
```

---

## 🔁 Dynamically Building Prompts

You can also create `PromptTemplate` from a string using:

```python
PromptTemplate.from_template("Tell me a joke about {topic}")
```

Or generate prompt templates on-the-fly by loading them from `.txt` files or databases.

---

## ✅ Best Practices

| Practice                            | Why it Matters                          |
| ----------------------------------- | --------------------------------------- |
| Keep templates clear and concise    | Reduces LLM confusion                   |
| Avoid overloading variables         | Too many dynamic inputs = brittle       |
| Use defaults carefully              | Avoid unfilled slots                    |
| Store and version prompts           | Enables testing and rollback            |
| Use comments (in multiline prompts) | Improves readability for long templates |

---

## 🔚 Summary

| Feature                | Supported  |
| ---------------------- | ---------- |
| Dynamic placeholders   | ✅          |
| Multi-variable prompts | ✅          |
| Few-shot prompts       | ✅          |
| Reusable formatting    | ✅          |
| Chat model support     | ✅          |
| File/DB loading        | ✅ (custom) |

---

## 🧪 Mini Project Idea

> 🔨 *Build a prompt playground where users can enter variables and see the formatted prompt in real time.*

Let me know if you'd like a template or tool to help you do this.

---
