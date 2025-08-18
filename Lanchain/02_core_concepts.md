Here's a detailed explanation of the **core concepts** of **LangChain**, focusing on **Chains, Agents, Tools, and Prompts** — the building blocks of any intelligent LangChain application.

---

## 🧠 **02\_understanding\_core\_concepts.md**

> 📌 *Topic: Understand the core concepts – Chains, Agents, Tools, and Prompts*

---

## 1. 🔗 **Chains**

### ✅ What is a Chain?

A **Chain** is a **sequence of steps** where each step can involve calling an LLM, performing an action, or combining inputs/outputs.

LangChain uses Chains to **orchestrate multi-step workflows**.

---

### 🛠️ Common Types of Chains:

| Chain Type              | Description                       |
| ----------------------- | --------------------------------- |
| `LLMChain`              | Basic chain with prompt + LLM     |
| `SimpleSequentialChain` | Executes steps one after another  |
| `SequentialChain`       | Executes steps with shared memory |
| `RetrievalQA`           | Integrates vector search with LLM |

---

### 💡 Example:

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("AI tools"))
```

---

## 2. 🤖 **Agents**

### ✅ What is an Agent?

An **Agent** is a component that **decides what actions to take**, step by step, using the output of an LLM.

Agents are **autonomous decision-makers**. They can use Tools to answer user queries in multiple steps.

---

### 🚀 Example Use Case:

> “What’s the weather in Gorakhpur and convert it to Fahrenheit?”

The agent will:

* Use a **weather API tool**
* Use a **calculator tool**
* Return the answer

---

### 🔥 Common Agent Types:

| Agent Type      | Description                                  |
| --------------- | -------------------------------------------- |
| `ZeroShotAgent` | Uses prompt examples to act                  |
| `ReActAgent`    | Reasoning and Acting with intermediate steps |
| `ChatAgent`     | Conversational agent with tools              |
| `Custom Agents` | You define behavior                          |

---

## 3. 🛠️ **Tools**

### ✅ What is a Tool?

A **Tool** is a wrapper around a **function or API** that agents can use — like a search engine, calculator, or a custom Python function.

---

### 🔨 Example Tool:

```python
from langchain.tools import tool

@tool
def multiply_by_two(number: int) -> int:
    return number * 2
```

You can then register this tool and let the agent use it during reasoning.

---

### 🔌 Built-in Tools:

| Tool          | Purpose            |
| ------------- | ------------------ |
| `SerpAPI`     | Google search      |
| `Calculator`  | Math calculations  |
| `Python REPL` | Code execution     |
| `SQLDatabase` | Querying databases |
| `Requests`    | HTTP requests      |

---

## 4. 🧾 **Prompts**

### ✅ What is a Prompt?

A **Prompt** is a **text template** that guides the LLM in generating a specific type of output.

LangChain provides structured `PromptTemplate` and `ChatPromptTemplate` classes for dynamic prompt generation.

---

### 🧩 Components of PromptTemplate:

* **Input variables**
* **Template string**

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
```

---

### 🧠 Best Practices:

* Keep prompts clear and consistent
* Use few-shot examples if necessary
* Experiment with temperature and max tokens

---

## 🔁 Putting It All Together

Imagine building a **Conversational Agent** that:

1. Uses **PromptTemplate** to structure input
2. Uses a **Chain** to process user data
3. Uses **Memory** to hold conversation history
4. Lets an **Agent** decide which **Tool** to use
5. Finally calls an **LLM** to generate the response

This is **LangChain in action**!

---

## ✅ Summary Table

| Concept    | Description                   | Example                       |
| ---------- | ----------------------------- | ----------------------------- |
| **Chain**  | Multi-step LLM workflow       | Summarize → Translate → Store |
| **Agent**  | Makes decisions using tools   | Web search + math + response  |
| **Tool**   | Functional interface for APIs | Google, Calculator, SQL       |
| **Prompt** | Structured query to the LLM   | “What is a good name for…”    |

---

Let me know if you want this in Markdown file format or combined with other lessons in a ZIP for your series.
