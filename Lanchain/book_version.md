
---

## 📘 **What is LangChain?**

LangChain is an **open-source framework** designed to help developers build **powerful applications using Large Language Models (LLMs)**. It provides modular components to simplify the integration of LLMs with:

* External data (documents, APIs, databases)
* Reasoning capabilities (tools, agents)
* Memory (for context-aware conversations)

LangChain makes it easier to go from a simple prompt → to a complex application.

---

## 🎯 **Purpose of LangChain**

LangChain is not just a wrapper for LLMs like GPT-4 — it’s a **toolkit to build intelligent agents and pipelines** that interact with data and make decisions.

### Key Goals:

* **Chain complex LLM calls**: Break workflows into logical steps.
* **Enable memory and context**: Keep conversation or task history.
* **Integrate tools and APIs**: Use external tools like search engines, calculators, and databases.
* **Handle documents**: Load, chunk, index, and query documents with LLMs.

---

## 🏗️ **LangChain Architecture**

LangChain is built around the idea of **composable components**. Here are the core building blocks:

### 1. **LLMs**

Abstractions over models like OpenAI, Cohere, Hugging Face, etc.

### 2. **Prompts**

Structured templates for generating dynamic prompts with variable injection.

### 3. **Chains**

Orchestrate calls to LLMs. Types include:

* `LLMChain`: Basic single-step call
* `SequentialChain`: Multiple chained steps
* `RetrievalQA`: Uses vector store retrieval + LLM

### 4. **Memory**

Add memory to chains for conversation continuity.

Examples:

* `ConversationBufferMemory`
* `SummaryMemory`
* `EntityMemory`

### 5. **Agents**

Use LLMs to **reason about which tools to use**. They act like autonomous decision-makers.

### 6. **Tools**

External functions the agent can use — calculators, search APIs, etc.

### 7. **Document Loaders & Text Splitters**

* Loaders: Read PDFs, URLs, JSON, Notion, etc.
* Splitters: Divide text into manageable chunks

### 8. **Embeddings & Vector Stores**

* Generate semantic embeddings
* Store and retrieve using FAISS, Pinecone, Chroma, etc.

---

## 🌐 **Ecosystem Overview**

LangChain has a rapidly growing ecosystem of integrations and community tools:

| Component         | Examples / Integrations                    |
| ----------------- | ------------------------------------------ |
| **LLMs**          | OpenAI, Anthropic, Hugging Face, Cohere    |
| **Embeddings**    | OpenAI, Hugging Face, SentenceTransformers |
| **Vector Stores** | FAISS, Pinecone, Chroma, Weaviate, Qdrant  |
| **Databases**     | SQL, MongoDB, Redis                        |
| **UI Tools**      | Streamlit, Gradio, React                   |
| **Memory**        | BufferMemory, SummaryMemory                |
| **Agents**        | ReAct, ZeroShot, ChatAgent, Custom agents  |

LangChain also supports:

* **LangServe** – For deploying chains as APIs
* **LangSmith** – Debugging and tracing LLM workflows

---

## 🧠 Example Use Cases

| Use Case                                 | Description                                             |
| ---------------------------------------- | ------------------------------------------------------- |
| **Chatbots**                             | Context-aware assistants with tools                     |
| **Document Q\&A**                        | Ask questions over large document sets                  |
| **Autonomous Agents**                    | Agents that search the web, take notes, make decisions  |
| **RAG (Retrieval-Augmented Generation)** | Combine vector DBs with LLMs for dynamic answers        |
| **LLM Apps**                             | Use LLMs in workflows, automations, or multi-step tasks |

---

## ✅ Summary

| Feature         | Description                                    |
| --------------- | ---------------------------------------------- |
| **Goal**        | Build LLM apps with context, memory, and logic |
| **Modularity**  | Components like chains, tools, agents          |
| **Flexibility** | Works with many LLMs, databases, and APIs      |
| **Power**       | Enables RAG, chatbots, data agents, more       |

LangChain simplifies complex AI application development by abstracting prompt engineering, chaining logic, memory handling, and tool integration — allowing you to focus on what matters: **building smarter AI apps**.

---
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


---

## 🧪 Prerequisites

Before starting, make sure you have:

| Requirement     | Notes                                |
| --------------- | ------------------------------------ |
| Python 3.8+     | Preferably 3.10+                     |
| pip             | Python package manager               |
| virtualenv/venv | For isolated environments            |
| OpenAI API key  | For LLM access (optional but common) |

---

## 🔧 Step-by-Step Setup

### 🔹 1. **Create and activate a virtual environment**

```bash
# Create project folder
mkdir langchain_app && cd langchain_app

# Create virtual environment
python -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

---

### 🔹 2. **Install LangChain and dependencies**

```bash
pip install langchain openai tiktoken
```

> ✅ `openai`: for using OpenAI LLMs
> ✅ `tiktoken`: for token counting (prompt size estimation)

---

### 🔹 3. **Install optional tools**

You can install additional packages depending on your use case:

| Use Case            | Install Command                                  |
| ------------------- | ------------------------------------------------ |
| Document loaders    | `pip install unstructured pdfminer.six`          |
| Vector DBs          | `pip install faiss-cpu pinecone-client chromadb` |
| Web UI (Streamlit)  | `pip install streamlit`                          |
| Tracing & Debugging | `pip install langsmith`                          |

---

### 🔹 4. **Set your API keys (e.g. OpenAI)**

LangChain accesses models via environment variables.

Create a `.env` file in your root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Or set it directly:

```bash
export OPENAI_API_KEY=your_openai_api_key_here   # Linux/macOS
set OPENAI_API_KEY=your_openai_api_key_here      # Windows
```

Use `python-dotenv` to load `.env` automatically:

```bash
pip install python-dotenv
```

Then in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🚀 Quick Test Example

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
response = llm("Explain LangChain in 1 sentence.")
print(response)
```

If everything is set up correctly, you should see an LLM-generated sentence.

---

## 📂 Suggested Project Structure

```
/langchain_app/
├── chains/
├── tools/
├── prompts/
├── main.py
├── .env
├── requirements.txt
```

---

## 📄 Save dependencies

Once your app is working, save your dependencies:

```bash
pip freeze > requirements.txt
```

---

## 🧼 Deactivate environment

When you're done:

```bash
deactivate
```

---

## ✅ Summary

| Step                      | Command / Action               |
| ------------------------- | ------------------------------ |
| Create environment        | `python -m venv venv`          |
| Install LangChain         | `pip install langchain openai` |
| Add API keys              | Set `OPENAI_API_KEY` in `.env` |
| Run basic example         | `llm("Hello world")`           |
| Extend with tools/loaders | Install plugins as needed      |

---

---

## 🧠 1. Overview of Each Framework

| Framework                           | Description                                                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **LangChain**                       | Modular framework to build LLM-powered applications using agents, tools, chains, memory, and integrations |
| **LlamaIndex** (formerly GPT Index) | Framework focused on ingesting, indexing, and querying data using LLMs (RAG-centric)                      |
| **Haystack**                        | NLP framework built around pipelines and retrievers, more traditional, supports search + LLMs             |
| **Custom Pipelines**                | Fully manual development using LLM APIs, retrieval logic, embeddings, etc., without any framework         |

---

## ⚙️ 2. Architectural Philosophy

| Feature/Aspect            | **LangChain**         | **LlamaIndex**                     | **Haystack**     | **Custom Pipelines** |
| ------------------------- | --------------------- | ---------------------------------- | ---------------- | -------------------- |
| **Design Goal**           | Modular AI apps       | Indexing & RAG                     | NLP pipelines    | Total control        |
| **Main Abstraction**      | Chains, Agents, Tools | Index, Query Engine                | Nodes, Pipelines | Custom Code          |
| **LLM-Centric**           | ✅ Yes                 | ✅ Yes                              | Optional         | Depends              |
| **Data Ingestion**        | Basic loaders         | Advanced loaders & transformations | Good support     | Manual               |
| **Vector DB Integration** | ✅ Strong              | ✅ Strong                           | ✅ Strong         | Manual               |
| **Agents/Tools**          | ✅ Powerful            | ❌ None                             | ✅ Limited        | Custom required      |
| **Memory/Context**        | ✅ Built-in            | ✅ Context manager                  | ❌ (manual)       | Manual               |

---

## 🔧 3. Core Features Comparison

| Feature                          | LangChain                               | LlamaIndex      | Haystack       | Custom Pipelines |
| -------------------------------- | --------------------------------------- | --------------- | -------------- | ---------------- |
| Document Loaders                 | ✅                                       | ✅ (rich)        | ✅              | 🔧 You write it  |
| Text Splitters                   | ✅                                       | ✅               | ✅              | 🔧               |
| Vector Store Support             | ✅ (many: FAISS, Pinecone, Chroma, etc.) | ✅               | ✅              | 🔧               |
| Embedding Model Integration      | ✅ (HuggingFace, OpenAI, etc.)           | ✅               | ✅              | 🔧               |
| Retrieval-Augmented Generation   | ✅                                       | ✅               | ✅              | ✅                |
| Agents and Tools (ReAct, custom) | ✅ Powerful                              | ❌               | ⚠️ Limited     | 🔧               |
| Prompt Templates and Chains      | ✅ Modular                               | ✅ Prompt helper | ⚠️ Not primary | 🔧               |
| Tracing / Debugging              | ✅ LangSmith                             | ✅ (basic)       | ❌              | ❌                |
| UI Integration (Streamlit, etc.) | ✅ Easy                                  | ⚠️ Manual       | ✅ Streamlit UI | 🔧               |

---

## 🚀 4. When to Use Which?

### ✅ **Use LangChain If:**

* You need **agent-based reasoning**
* You want to integrate **tools and APIs** (e.g. Google search, calculator)
* You're building **LLM-first apps** with memory and chaining
* You want **prompt engineering control**

---

### ✅ **Use LlamaIndex If:**

* Your focus is **document ingestion, indexing, and querying**
* You're building **RAG pipelines** with minimal boilerplate
* You need to handle **structured or semi-structured data** (Pandas, SQL, Notion, etc.)
* You want to combine multiple **data connectors** easily

---

### ✅ **Use Haystack If:**

* You’re from a **traditional NLP background** (pre-LLM) and want **retrievers, rankers, readers**
* You want to use **OpenSearch, ElasticSearch**, or **non-LLM** QA models
* You’re building production-level NLP systems (question answering, summarization, etc.) with classical ML/LLM combo

---

### ✅ **Use Custom Pipelines If:**

* You want **maximum control** and minimal dependencies
* You’re an expert in handling LLM APIs, embeddings, memory
* You need to deeply optimize or secure your pipeline
* You don’t want to be locked into a specific framework

---

## ⚖️ 5. Pros and Cons

| Framework      | ✅ Pros                                           | ❌ Cons                                  |
| -------------- | ------------------------------------------------ | --------------------------------------- |
| **LangChain**  | Modular, agents/tools, memory, RAG, ecosystem    | Can feel complex for simple tasks       |
| **LlamaIndex** | Easy RAG, flexible loaders, fast iteration       | Not focused on agents/tools, limited UI |
| **Haystack**   | Production-ready NLP pipelines, great retrievers | LLM integration feels secondary         |
| **Custom**     | Full flexibility, no bloat                       | High maintenance, reinvents the wheel   |

---

## 🧪 Example Use Case Mapping

| Use Case                          | Best Tool           |
| --------------------------------- | ------------------- |
| Build an AI chatbot with tools    | **LangChain**       |
| Query 1000 PDFs via vector DB     | **LlamaIndex**      |
| Traditional QA system (BERT etc.) | **Haystack**        |
| Fully custom data + model setup   | **Custom pipeline** |

---

## ✅ Conclusion

| Choose...      | If you need...                          |
| -------------- | --------------------------------------- |
| **LangChain**  | LLM-first modular app with agents/tools |
| **LlamaIndex** | Easy RAG from many data sources         |
| **Haystack**   | Classical NLP or hybrid LLM pipelines   |
| **Custom**     | Full control and optimization           |

---

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

---

## 🧠 **08\_using\_memory\_in\_langchain.md**

> 📌 *Topic: Implement memory with `ConversationBufferMemory`, `ConversationSummaryMemory`, and more*

---

## 🧾 What is Memory in LangChain?

In LangChain, **Memory** is a component that lets chains and agents **remember previous interactions** or conversation state.

Without memory, your chatbot or assistant will forget the context of the previous messages.

LangChain provides various memory types based on your application needs.

---

## 🧩 Types of Memory

| Memory Type                               | Description                               |
| ----------------------------------------- | ----------------------------------------- |
| `ConversationBufferMemory`                | Stores full conversation history as-is    |
| `ConversationSummaryMemory`               | Summarizes the conversation so far        |
| `ConversationBufferWindowMemory`          | Keeps a sliding window of recent messages |
| `ConversationTokenBufferMemory`           | Keeps recent tokens under a limit         |
| `VectorStoreRetrieverMemory`              | Stores memory in a vector database        |
| `ZepMemory`, `PostgresMemory` (3rd-party) | Persistent/structured memory backends     |

---

## 🔗 Memory Components Structure

Each memory type works by attaching to a **chain or agent**, usually along with a `ChatPromptTemplate` or `LLMChain`.

---

## ✅ 1. Using `ConversationBufferMemory`

### 📦 What It Does:

Stores all messages (user + AI) **verbatim**. Useful for small conversations.

### 🧪 Code Example:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

memory = ConversationBufferMemory()
llm = OpenAI(temperature=0.7)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response = conversation.predict(input="Hi, I'm Sandeep.")
print(response)

response = conversation.predict(input="What did I just tell you?")
print(response)
```

---

## ✅ 2. Using `ConversationSummaryMemory`

### 📦 What It Does:

Summarizes the conversation to save memory space and reduce token usage. Good for **longer chats**.

### 🧪 Code Example:

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="I want to learn Python.")
conversation.predict(input="Can you explain variables?")
```

> 💡 Internally, it creates a summary like:
> "The user wants to learn Python. They asked about variables."

---

## ✅ 3. Using `ConversationBufferWindowMemory`

### 📦 What It Does:

Stores **only the last N messages** (configurable). Useful when you want recency and to avoid token overflow.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="Tell me about LangChain.")
conversation.predict(input="Who created it?")
conversation.predict(input="What is memory?")
```

> Only the last 2 exchanges will be retained in memory.

---

## ✅ 4. Using `ConversationTokenBufferMemory`

### 📦 What It Does:

Keeps only as many tokens as allowed (e.g., 1000 tokens). Useful when using **token-limited models**.

```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=1000)
conversation = ConversationChain(llm=llm, memory=memory)
```

---

## ✅ 5. Using `VectorStoreRetrieverMemory`

### 📦 What It Does:

Stores past messages as **embeddings in a vector store** and retrieves the most relevant ones during the conversation.

> Useful when:
> ✅ Long-running sessions
> ✅ Need semantic search over conversation history

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

embedding = OpenAIEmbeddings()
vectorstore = FAISS(embedding_function=embedding)

retriever_memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever())

conversation = ConversationChain(
    llm=llm,
    memory=retriever_memory,
    verbose=True
)
```

---

## 📁 Sample Conversation Output

```text
User: My name is Sandeep.
AI: Nice to meet you, Sandeep!
User: What did I tell you?
AI: You told me your name is Sandeep.
```

With memory, the model **recalls** context and maintains continuity.

---

## ⚙️ Memory + Agents

You can also attach memory to **agents**:

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

> This is essential for multi-turn, tool-using agents like GPT-4-style assistants.

---

## 🧠 Which Memory to Use?

| Scenario                      | Best Memory Type                 |
| ----------------------------- | -------------------------------- |
| Short, simple chats           | `ConversationBufferMemory`       |
| Long chats with summarization | `ConversationSummaryMemory`      |
| Recency-based recall          | `ConversationBufferWindowMemory` |
| Token-limited models          | `ConversationTokenBufferMemory`  |
| Smart recall from embeddings  | `VectorStoreRetrieverMemory`     |

---

## ✅ Summary

| Memory Type                  | Key Feature                     |
| ---------------------------- | ------------------------------- |
| `ConversationBufferMemory`   | Full raw history                |
| `SummaryMemory`              | Summarized history              |
| `WindowMemory`               | Last N exchanges                |
| `TokenBufferMemory`          | Based on token count            |
| `VectorStoreRetrieverMemory` | Semantic search in conversation |

---

---

## 🤝 **09\_integrating\_llms\_openai\_huggingface\_cohere.md**

> 📌 *Topic: Integrate OpenAI, Hugging Face, Cohere, and other LLMs in LangChain*

---

## 🎯 Why Use Multiple LLM Providers?

| Reason                  | Benefit                                       |
| ----------------------- | --------------------------------------------- |
| 💵 Cost control         | Choose cheaper models for less critical tasks |
| ⚡ Performance tuning    | Pick faster models where latency matters      |
| 🧠 Model specialization | Some models perform better on certain domains |
| 💬 Fallbacks & backups  | Switch if one provider is unavailable         |
| 🔌 Hybrid architectures | Use multiple models in a single workflow      |

---

## 📦 1. Install Required Packages

```bash
pip install langchain openai cohere transformers huggingface_hub
```

---

## 🔑 2. Set API Keys

You can either export environment variables:

```bash
export OPENAI_API_KEY=your-openai-key
export COHERE_API_KEY=your-cohere-key
export HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
```

Or use a `.env` file with `python-dotenv`.

---

## 🧠 3. OpenAI Integration

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
response = llm("What is LangChain?")
print(response)
```

You can configure:

* `model_name`: e.g., `"gpt-3.5-turbo"`, `"gpt-4"`
* `max_tokens`, `temperature`, etc.

---

## 🤗 4. Hugging Face Integration (via `HuggingFaceHub`)

```python
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 100}
)

print(llm("Explain neural networks in simple terms"))
```

> Hugging Face supports both **transformer models** (like T5, GPT2) and **inference API** (paid tier).

---

## 🧬 5. Hugging Face Local Model (via `HuggingFacePipeline`)

If you want to **run models locally**, use:

```python
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

print(llm("Write a short poem about a sunrise."))
```

> ⚠️ Local models require more RAM/GPU.

---

## 🧪 6. Cohere Integration

```python
from langchain.llms import Cohere

llm = Cohere(
    model="command-nightly",
    temperature=0.5,
    max_tokens=100
)

print(llm("Summarize the history of India."))
```

> Cohere is fast and focused on business-ready LLMs.

---

## 🔄 7. Switching LLMs in a Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Translate to French: {sentence}")
sentence = "I love AI."

# Using OpenAI
openai_chain = LLMChain(llm=OpenAI(), prompt=prompt)
print(openai_chain.run(sentence=sentence))

# Using Cohere
cohere_chain = LLMChain(llm=Cohere(), prompt=prompt)
print(cohere_chain.run(sentence=sentence))
```

> You can switch `llm=` dynamically in your pipeline!

---

## ⚙️ 8. Building a Multi-Model Agent

You can even assign different tasks to different models.

```python
summarizer = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo"),
    prompt=PromptTemplate.from_template("Summarize: {text}")
)

translator = LLMChain(
    llm=HuggingFaceHub(repo_id="t5-base"),
    prompt=PromptTemplate.from_template("Translate to German: {text}")
)
```

---

## 📊 LLM Provider Comparison

| Provider         | Pros                         | Cons                           |
| ---------------- | ---------------------------- | ------------------------------ |
| **OpenAI**       | Top-tier quality, chat modes | Cost, rate limits              |
| **Hugging Face** | Flexible, many models        | May need paid API or local GPU |
| **Cohere**       | Fast, cheaper                | Fewer model types              |
| **Local (HF)**   | Full control, privacy        | Resource intensive             |

---

## ✅ Summary

| LLM Type          | Class in LangChain      |
| ----------------- | ----------------------- |
| OpenAI            | `langchain.llms.OpenAI` |
| HuggingFace Hub   | `HuggingFaceHub`        |
| HuggingFace Local | `HuggingFacePipeline`   |
| Cohere            | `Cohere`                |

---

## 🚀 Next Steps

* Use different LLMs in **different chains** (e.g., summarization via OpenAI, sentiment via Cohere)
* Add **fallbacks** if one provider fails
* Tune **temperature, max tokens, and prompts** per provider

---

---

## 🤖 **10\_introduction\_to\_agents\_zero\_shot\_react\_custom\_tools.md**

> 📌 *Topic: Introduction to Agents: `ZeroShotAgent`, `ReAct`, and Creating Custom Tools*

---

## 🧠 What is an Agent in LangChain?

An **Agent** is a special component in LangChain that can:

* Use **reasoning** and **tool selection** to decide what action to take next
* Dynamically choose between multiple **tools** (APIs, functions, calculators, web search, etc.)
* Handle **multi-step tasks** with memory and context

> 🧠 Agents = LLMs that can **think + act + observe + decide again**

---

## ⚙️ Core Components of Agents

| Component  | Role                                         |
| ---------- | -------------------------------------------- |
| **Agent**  | The decision-maker using an LLM              |
| **Tools**  | Callable functions the agent can choose from |
| **LLM**    | The language model used for reasoning        |
| **Prompt** | Guides the LLM on how to behave and choose   |
| **Memory** | (Optional) Maintains context across steps    |

---

## 🚀 1. `ZeroShotAgent` – Use Tools with No Prior Examples

### 📦 Description:

* Uses a **descriptive prompt** instead of training examples
* Relies on the LLM's ability to "figure it out" using the tool names + descriptions
* Ideal for **simple tool orchestration**

---

### 🧪 Example:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import requests

# Define a simple tool
def get_weather(city):
    return f"The weather in {city} is sunny."  # Mocked response

tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="Returns current weather in a given city"
    )
]

llm = OpenAI(temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

agent.run("What's the weather in Paris?")
```

---

## 🧠 2. ReAct Agent – Reason + Act (Used Behind `zero-shot-react-description`)

### 📦 What is ReAct?

* A prompting strategy: **Reasoning + Acting + Observing**
* The agent uses LLM to think step-by-step, call a tool, and then observe the output before moving on.

Example output of ReAct Agent:

```
Thought: I need to find the weather for Paris.
Action: get_weather
Action Input: Paris
Observation: The weather in Paris is sunny.
Final Answer: It is sunny in Paris.
```

> You don't need to configure it directly — `zero-shot-react-description` uses ReAct under the hood.

---

## 🛠️ 3. Creating Custom Tools

You can define tools as simple functions or classes.

### 🧪 Example: Calculator Tool

```python
def calculator_tool(input_str: str):
    try:
        return str(eval(input_str))
    except:
        return "Invalid math expression"

tool = Tool(
    name="calculator",
    func=calculator_tool,
    description="Performs basic arithmetic operations like 2+2"
)
```

You can now add this to your agent’s tool list.

---

## ⚠️ Tool Design Best Practices

| Rule                             | Why it Matters                   |
| -------------------------------- | -------------------------------- |
| Use simple input/output (string) | Easier for LLMs to interact with |
| Add clear `description`          | Helps LLM choose the right tool  |
| Keep tools atomic                | One job per tool                 |
| Handle errors gracefully         | LLMs don’t like broken tools     |

---

## 🧠 Advanced Use Cases

| Use Case               | Agent Type               | Tools Used                       |
| ---------------------- | ------------------------ | -------------------------------- |
| Web browsing assistant | ReAct Agent              | WebSearchTool, BrowserTool       |
| Data analysis bot      | ZeroShotAgent            | Python REPL, CalculatorTool      |
| RAG chatbot            | ConversationalReActAgent | VectorRetrieverTool, Memory      |
| DevOps LLM helper      | Tool-calling Agent       | BashTool, GitTool, DocSearchTool |

---

## 💡 When NOT to Use Agents

Use **basic chains** instead of agents if:

* The flow is **fixed or predictable**
* You don’t need decision-making or dynamic tool usage
* You want **speed** and **cost-efficiency**

---

## ✅ Summary Table

| Feature             | ZeroShotAgent      | ReActAgent     | Custom Chains        |
| ------------------- | ------------------ | -------------- | -------------------- |
| Reasoning           | ✅ Basic via prompt | ✅ Step-by-step | ❌ Pre-defined only   |
| Dynamic Tool Use    | ✅ Yes              | ✅ Yes          | ❌                    |
| Tool Integration    | ✅ Easy             | ✅ Required     | ✅                    |
| Memory Integration  | ✅ Optional         | ✅ Optional     | ✅                    |
| Custom Tool Support | ✅ via `Tool()`     | ✅ via `Tool()` | ✅ via function calls |

---

## 🔚 Final Thoughts

Agents are ideal when:

* Your app needs **multi-step reasoning**
* You want to let the LLM **choose its own path**
* You need **tool use with flexibility**

---

---

## 📄 **11\_loading\_data\_from\_pdfs\_notion\_csv\_webpages.md**

> 📌 *Topic: Load data from PDFs, Notion, CSV, and web pages using built-in loaders*

---

## 📦 Why Use Loaders?

LangChain provides **`DocumentLoader`** classes to easily ingest data from various formats into a unified `Document` structure.

Each loader returns a list of `Document` objects like:

```python
Document(
    page_content="text",
    metadata={"source": "file name", "page": 3}
)
```

---

## ⚙️ 1. Installing Loaders

```bash
pip install langchain unstructured pdfminer.six beautifulsoup4 requests python-docx html2text
```

> Also required: API tokens for Notion, if applicable.

---

## 📚 2. Load PDF Files

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("files/myfile.pdf")
docs = loader.load()

print(docs[0].page_content[:300])  # Preview content
```

### 🧠 Other PDF loaders:

* `PDFMinerLoader`
* `PDFPlumberLoader`
* `PyMuPDFLoader`

---

## 🗂️ 3. Load CSV Files

Each **row** becomes a `Document`.

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader(file_path="files/data.csv")
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
```

✅ Supports large CSVs. You can customize field parsing with `encoding`, `delimiter`, etc.

---

## 📝 4. Load Notion Pages (via API)

### Step 1: Install required dependencies

```bash
pip install notion-client
```

### Step 2: Load documents from Notion

```python
from langchain.document_loaders import NotionDBLoader

loader = NotionDBLoader(
    integration_token="your_notion_api_key",
    database_id="your_notion_db_id"
)

docs = loader.load()
```

> 📌 Get your [Notion API Key & DB ID](https://developers.notion.com/docs/getting-started)

---

## 🌐 5. Load Web Pages (HTML URLs)

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")
docs = loader.load()

print(docs[0].page_content[:300])
```

> Uses `requests` + `BeautifulSoup` internally
> ✅ Great for scraping blogs, docs, wiki pages

---

## 🖼️ 6. Load from Markdown or Text

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("files/notes.txt", encoding='utf-8')
docs = loader.load()
```

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("files/notes.md")
docs = loader.load()
```

---

## 📸 7. Load Images (OCR with Tesseract)

```bash
pip install pytesseract pillow
```

```python
from langchain.document_loaders import UnstructuredImageLoader

loader = UnstructuredImageLoader("images/screenshot.png")
docs = loader.load()
```

> Useful for **scanned PDFs**, **handwritten notes**, or **screenshots**

---

## 🧠 Bonus: Combine Multiple Sources

```python
from langchain.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader

pdf_docs = PyPDFLoader("sample.pdf").load()
csv_docs = CSVLoader(file_path="data.csv").load()
web_docs = WebBaseLoader("https://example.com").load()

all_docs = pdf_docs + csv_docs + web_docs
```

---

## 🧩 Next Step: Split Documents into Chunks

Once loaded, split large documents into chunks for processing:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs_chunked = splitter.split_documents(all_docs)
```

---

## ✅ Summary Table

| Source        | Loader Class                    | Notes                      |
| ------------- | ------------------------------- | -------------------------- |
| **PDF**       | `PyPDFLoader`, `PDFMinerLoader` | Extracts text page-wise    |
| **CSV**       | `CSVLoader`                     | One row = one document     |
| **Notion**    | `NotionDBLoader`                | Needs API token + DB ID    |
| **Web pages** | `WebBaseLoader`                 | HTML to text using `bs4`   |
| **Images**    | `UnstructuredImageLoader`       | OCR-based text from images |
| **Text/MD**   | `TextLoader`, `MarkdownLoader`  | Loads raw text             |

---


---

## ❓ Why Split Documents?

Large documents:

* Can exceed the **token limit** of LLMs
* Are inefficient to search or embed
* Lack granular context control

✅ Splitting documents into **overlapping, semantically coherent chunks** enables:

* Better **retrieval**
* Faster **inference**
* Higher **answer accuracy**

---

## 🔧 1. `RecursiveCharacterTextSplitter` (Most Common)

### 📦 Description:

* Smartly breaks text by trying **larger separators first**, then **smaller ones**.
* Ensures **coherent chunks**, unlike naïve splitting.

### ✅ Recommended for:

* PDFs, raw text, CSVs, HTML, etc.

---

### 🧪 Example:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_text(
    "LangChain is a framework for building LLM-powered applications. It supports chains, agents, tools..."
)

print(chunks[:2])
```

---

## 📝 2. `MarkdownTextSplitter`

### 📦 Description:

* Uses **Markdown headers** and structure (`#`, `##`, `-`, `1.`) to split logically.
* Maintains **semantic boundaries**.

### ✅ Recommended for:

* `.md` files, technical notes, documentation, Notion exports.

---

### 🧪 Example:

```python
from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=50)

markdown = """
# Introduction
LangChain is powerful.

## Features
- Agents
- Tools

## Installation
Use pip to install LangChain.
"""

chunks = splitter.split_text(markdown)
print(chunks)
```

---

## 🧱 3. Understanding `chunk_size` and `chunk_overlap`

| Parameter       | Meaning                                                              |
| --------------- | -------------------------------------------------------------------- |
| `chunk_size`    | Max number of characters per chunk (often \~500–1000 for LLMs)       |
| `chunk_overlap` | Number of characters shared between adjacent chunks (usually 50–150) |

> 🔁 **Overlap** ensures continuity and prevents context loss at chunk boundaries.

---

## 📚 4. Chunking a List of Documents

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents (from PDFs, CSVs, etc.)
split_docs = text_splitter.split_documents(documents)
print(f"Created {len(split_docs)} chunks.")
```

---

## 🧠 Best Practices for Splitting

| Best Practice                     | Why It Helps                                |
| --------------------------------- | ------------------------------------------- |
| Use **recursive splitters**       | Avoids mid-sentence or mid-paragraph breaks |
| Keep **chunk size ≤ 1000 tokens** | Compatible with most LLM input limits       |
| Use **chunk overlap ≥ 100**       | Preserves context across chunks             |
| Choose splitters based on format  | e.g. Markdown splitter for `.md`            |
| Test with real content            | Always validate quality of chunking output  |

---

## 📦 Optional: Custom Splitter for JSON/XML

You can subclass `TextSplitter` if you're working with custom formats:

```python
from langchain.text_splitter import TextSplitter

class MyCustomSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("<entry>")  # Example

# Use: MyCustomSplitter().split_text(...)
```

---

## ✅ Summary Table

| Splitter                           | Best For               | Strategy Used             |
| ---------------------------------- | ---------------------- | ------------------------- |
| `RecursiveCharacterTextSplitter`   | General text, PDFs     | Smart separator recursion |
| `MarkdownTextSplitter`             | Markdown, docs, Notion | Header-based structure    |
| `CharacterTextSplitter`            | Simple baseline use    | Naïve character chunks    |
| `SentenceTransformersTextSplitter` | Embedding-aware splits | (Coming soon/Custom)      |

---

## 🧪 Bonus: Visualize Your Chunked Data

```python
for i, doc in enumerate(split_docs[:3]):
    print(f"Chunk {i+1}:\n", doc.page_content[:300], "\n")
```

---

---

## 🧠 What Are Embeddings?

**Embeddings** are dense vector representations of text. They allow LLM apps to:

* Perform **semantic search**
* Enable **contextual recall**
* Power **RAG (Retrieval-Augmented Generation)** systems

LangChain supports multiple **embedding models** and **vectorstores** out of the box.

---

## 🧾 Step-by-Step Workflow

1. Load and split documents
2. Generate embeddings
3. Store in a vectorstore (FAISS, Pinecone, etc.)
4. Query for relevant documents

---

## ⚙️ 1. Install Required Packages

```bash
pip install langchain openai faiss-cpu pinecone-client chromadb weaviate-client
```

---

## 🧬 2. Choose an Embedding Model

LangChain supports:

```python
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
```

Other options:

* `HuggingFaceEmbeddings`
* `CohereEmbeddings`
* `GooglePalmEmbeddings`
* `OllamaEmbeddings` (for local use)

---

## 🗃️ 3. Store in FAISS (Local, Fast, Lightweight)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Prepare documents
docs = ["What is LangChain?", "LangChain is a framework for LLM apps."]
docs = [Document(page_content=doc) for doc in docs]

# Create FAISS index
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)

# Save locally
db.save_local("faiss_index")

# Load it back later
db = FAISS.load_local("faiss_index", embedding)

# Query
results = db.similarity_search("Tell me about LangChain", k=2)
print(results[0].page_content)
```

---

## 🌲 4. Store in Chroma (Local, Lightweight + Persistent)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
chroma_db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")

# Persist to disk
chroma_db.persist()

# Reload later
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# Query
docs = chroma_db.similarity_search("LangChain framework", k=1)
print(docs[0].page_content)
```

---

## 🌐 5. Store in Pinecone (Cloud-based, Scalable)

### 🔐 Set API Keys

```bash
export PINECONE_API_KEY=your_key
```

### 🧪 Code

```python
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

pinecone.init(api_key="your_key", environment="us-east1-gcp")

index_name = "langchain-demo"
embedding = OpenAIEmbeddings()

# Upload to Pinecone
doc_store = Pinecone.from_documents(docs, embedding, index_name=index_name)

# Query
results = doc_store.similarity_search("framework for LLM", k=1)
print(results[0].page_content)
```

> ⚠️ Pinecone charges for storage and usage — best for production-scale apps.

---

## 🧠 6. Store in Weaviate (Self-hosted or Cloud, Schema-based)

### 🔐 Prerequisite: Run Weaviate locally or use SaaS

```python
import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings

client = weaviate.Client("http://localhost:8080")  # or SaaS URL

embedding = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(
    docs,
    embedding,
    client=client,
    index_name="LangDocs",
    by_text=False
)

results = vectorstore.similarity_search("What is LangChain?", k=1)
print(results[0].page_content)
```

> ✅ Weaviate also supports metadata filtering and hybrid search.

---

## 🔎 7. Querying VectorStores

Most vectorstores support:

```python
docs = vectorstore.similarity_search("my query", k=3)
for doc in docs:
    print(doc.page_content)
```

Also supported:

* `.similarity_search_by_vector()`
* `.max_marginal_relevance_search()`
* `.as_retriever()` (for integration with chains)

---

## ✅ Summary Comparison

| Vector Store | Type        | Best For                | Persistence | Cost   |
| ------------ | ----------- | ----------------------- | ----------- | ------ |
| **FAISS**    | Local       | Fast prototyping        | ✅ Yes       | Free   |
| **Chroma**   | Local       | Lightweight, persistent | ✅ Yes       | Free   |
| **Pinecone** | Cloud       | Scalable production use | ✅ Yes       | Paid   |
| **Weaviate** | Cloud/local | Semantic + metadata     | ✅ Yes       | Hybrid |

---

---

## 📦 What is RAG (Retrieval-Augmented Generation)?

**RAG = Retriever + Generator**

| Component | Function                                  |
| --------- | ----------------------------------------- |
| Retriever | Finds relevant docs using embeddings      |
| Generator | Uses an LLM to answer based on those docs |

✅ Useful for:

* Chatbots with knowledge base
* Context-aware Q\&A over PDFs, sites
* Internal documentation search

---

## 📐 Architecture

```
User Query
    ↓
Retriever (FAISS / Chroma / Pinecone)
    ↓
Relevant Chunks (Text)
    ↓
Prompt + LLM (OpenAI, Cohere, etc.)
    ↓
Generated Answer
```

---

## 🔧 Setup: Install Requirements

```bash
pip install langchain openai faiss-cpu chromadb tiktoken pdfminer.six
```

---

## 🪵 Step 1: Load + Split Your Documents

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("your_file.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
```

---

## 🧬 Step 2: Generate Embeddings + Store in FAISS

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)
```

---

## 🔎 Step 3: Create a Retriever

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

Optional:

```python
# For MMR (diverse results)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
```

---

## 💬 Step 4: Setup the Generative Model (LLM)

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.3)
```

---

## 🔗 Step 5: Build the RetrievalQA Chain

```python
from langchain.chains import RetrievalQA

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # or map_reduce, refine
)

# Use it
query = "What is LangChain and how does it help with LLMs?"
result = rag_chain.run(query)

print("Answer:", result)
```

---

## ✅ You Now Have a RAG System!

### Example output:

```
Answer: LangChain is a framework designed to simplify the development of applications powered by large language models (LLMs). It offers tools for chaining together components like prompts, LLMs, retrievers, and memory.
```

---

## 🧩 Optional Enhancements

### 🧠 Add Memory to the RAG Chain

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
rag_chain.memory = memory
```

---

### 🎯 Streamlit UI

```python
# Save as app.py
import streamlit as st

st.title("📚 RAG Chatbot")

user_query = st.text_input("Ask a question about the document")

if user_query:
    response = rag_chain.run(user_query)
    st.write(response)
```

Run it:

```bash
streamlit run app.py
```

---

## ⚙️ Advanced Options

| Option                    | Use-case                               |
| ------------------------- | -------------------------------------- |
| `chain_type="map_reduce"` | Better for long docs                   |
| `retriever=MMR`           | Diverse, less redundant chunks         |
| `metadata_filter`         | Filter results by tags, filename, etc. |
| `StreamingOutput`         | Real-time token stream to UI           |

---

## ✅ Summary

| Step     | Tool/Component                   |
| -------- | -------------------------------- |
| Load     | `PyPDFLoader`, `TextLoader`      |
| Split    | `RecursiveCharacterTextSplitter` |
| Embed    | `OpenAIEmbeddings`               |
| Store    | `FAISS`, `Chroma`, etc.          |
| Retrieve | `.as_retriever()`                |
| Generate | `OpenAI`, `Cohere`, etc.         |
| Chain    | `RetrievalQA.from_chain_type()`  |

---

---

## 🧠 **15\_building\_intelligent\_chatbot\_with\_agents\_tools\_memory.md**

> 📌 *Topic: Create an intelligent chatbot using memory, tools, and agents in LangChain*

---

## 🧾 What You’ll Build

An AI assistant that:

* ✅ Understands user intent
* ✅ Uses **tools** (e.g. calculator, search, weather)
* ✅ Has **memory** of the conversation
* ✅ Uses **agents** to decide what to do

---

## 📦 Requirements

```bash
pip install langchain openai
```

> Optional (for tools like weather, web scraping, etc.):

```bash
pip install requests beautifulsoup4
```

---

## 🔧 Step 1: Setup LLM

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
```

---

## 🧰 Step 2: Create Tools

Tools are functions wrapped with metadata so agents can call them.

### ➕ Example: Calculator Tool

```python
from langchain.agents import Tool

def calculator(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for math expressions like '12 * 3'"
)
```

### 🌦️ Example: Weather (Mock)

```python
def weather_tool(city: str) -> str:
    return f"The weather in {city} is currently sunny, 29°C."  # Mock response

weather = Tool(
    name="Weather",
    func=weather_tool,
    description="Gives weather information for a city"
)
```

---

## 🧠 Step 3: Setup Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

---

## 🤖 Step 4: Create the Agent

```python
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

agent = initialize_agent(
    tools=[calc_tool, weather],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

---

## 💬 Step 5: Interact with the Chatbot

```python
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.run(user_input)
    print("Bot:", response)
```

---

### 🧪 Example Interaction:

```
You: What’s the weather in Mumbai?
Bot: The weather in Mumbai is currently sunny, 29°C.

You: What is 23 * 12?
Bot: 276

You: What did I ask you first?
Bot: You asked about the weather in Mumbai.
```

🎉 Your chatbot now:

* ✅ Remembers conversation
* ✅ Uses external tools
* ✅ Thinks before responding

---

## ⚙️ Bonus: Add More Tools

| Tool           | Functionality                  |
| -------------- | ------------------------------ |
| Web search     | Real-time info (requires API)  |
| WikipediaTool  | Inbuilt knowledge base         |
| Python REPL    | Run Python code safely         |
| Docs Retriever | Query internal documents (RAG) |

---

## 🧠 Summary

| Feature      | Used Component                    |
| ------------ | --------------------------------- |
| LLM          | `OpenAI` or other model           |
| Memory       | `ConversationBufferMemory`        |
| Reasoning    | `AgentType.CHAT_CONVERSATIONAL_*` |
| Tools        | `Tool(name, func, description)`   |
| Control Flow | `initialize_agent()`              |

---

---

## 🎯 Final Result

An intelligent chatbot that:

* Uses **LangChain agents**
* Remembers conversations
* Uses tools (e.g. calculator, mock weather)
* Runs in your browser via **Streamlit**

---

## 🧱 Project Structure

```
📁 langchain_chatbot/
├── app.py             👈 Main Streamlit app
├── tools.py           👈 Reusable tool functions
├── requirements.txt   👈 Dependencies
```

---

## 📦 Step 1: requirements.txt

```txt
streamlit
langchain
openai
```

> Install with:

```bash
pip install -r requirements.txt
```

---

## 🧰 Step 2: tools.py

```python
from langchain.agents import Tool

def calc_tool(input: str) -> str:
    try:
        return str(eval(input))
    except:
        return "Sorry, I couldn't compute that."

def weather_tool(city: str) -> str:
    return f"The weather in {city} is 30°C and sunny."  # Mock

calculator = Tool(
    name="Calculator",
    func=calc_tool,
    description="Useful for arithmetic expressions like '12 * 3'"
)

weather = Tool(
    name="Weather",
    func=weather_tool,
    description="Useful for checking the weather in a city"
)

tool_list = [calculator, weather]
```

---

## 🖥️ Step 3: app.py (Streamlit App)

```python
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from tools import tool_list

st.set_page_config(page_title="LangChain Chatbot", layout="centered")

# Header
st.title("🤖 LangChain Chatbot with Tools + Memory")

# API Key input
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")

if openai_api_key:
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tool_list,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )

    # Chat history in session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Type your message...")
    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)

        # Get response
        response = agent.run(user_input)
        st.chat_message("assistant").markdown(response)

        # Save conversation
        st.session_state.chat_history.append((user_input, response))

    # Render history
    for user, bot in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(bot)
else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
```

---

## ▶️ Step 4: Run the App

```bash
streamlit run app.py
```

---

## 🧠 Example Conversation

```
👤: What is 13 * 7?
🤖: 91

👤: What did I just ask?
🤖: You asked what 13 * 7 is.
```

---

## 🛠️ Optional Enhancements

| Feature                   | How to Add                                            |
| ------------------------- | ----------------------------------------------------- |
| RAG (PDF retrieval)       | Add `RetrieverTool` with vector DB (FAISS/Chroma)     |
| Chat with memory per user | Use `ConversationSummaryBufferMemory`                 |
| Upload file + QA          | Use `st.file_uploader`, parse → chunks → retriever    |
| Streaming response        | Use `st.empty()` and `LangchainCallbackHandler`       |
| Use local LLM             | Replace OpenAI with `Ollama` or `HuggingFacePipeline` |

---

## ✅ Summary

| Component   | Role                              |
| ----------- | --------------------------------- |
| `streamlit` | Web UI                            |
| `OpenAI`    | LLM for response generation       |
| `Tool`      | External logic (calc, weather)    |
| `Memory`    | Maintains context across messages |
| `Agent`     | Combines reasoning + tool calling |

---

---

## 💡 Why Build Custom Tools & Agents?

LangChain lets you:

* Extend capabilities by adding **domain-specific tools**
* Build **goal-oriented agents** with **multi-step reasoning**
* Solve business problems using **tool orchestration**

---

## 🧩 What You’ll Build

An agent that:

* Has multiple **custom tools** (calculator, docs retriever, date/time, web search, etc.)
* Uses **zero-shot reasoning** to choose the right tool
* Can be scaled into a modular, real-world assistant

---

## 🧱 Prerequisites

```bash
pip install langchain openai chromadb
```

---

## 🧰 1. Create and Register Your Own Tools

LangChain tools are just Python functions wrapped with metadata.

---

### 📘 Example 1: Calculator Tool

```python
from langchain.agents import Tool

def calculator_tool(query: str) -> str:
    try:
        return str(eval(query))
    except Exception:
        return "Error: Invalid math expression."

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Solves math problems like '23 * 4 + 9'"
)
```

---

### 📄 Example 2: Custom Document Search Tool (using Chroma)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Assume 'docs' is a list of Document objects (PDFs, etc.)
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

def search_docs(query: str) -> str:
    results = vectorstore.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])

doc_search_tool = Tool(
    name="DocumentSearch",
    func=search_docs,
    description="Searches internal documents to answer knowledge-based queries"
)
```

---

### 🕒 Example 3: Time Tool

```python
from datetime import datetime

def current_time(_: str = "") -> str:
    return f"Current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

time_tool = Tool(
    name="Time",
    func=current_time,
    description="Returns the current date and time"
)
```

---

## 🤖 2. Build a Custom Agent with Your Tools

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

tools = [calculator, doc_search_tool, time_tool]

llm = OpenAI(temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or CHAT_CONVERSATIONAL_REACT_DESCRIPTION
    verbose=True
)
```

---

## 💬 3. Interact with the Agent

```python
response = agent.run("What's 21 * 45 and what time is it?")
print(response)

response2 = agent.run("Can you find the summary from the onboarding PDF?")
print(response2)
```

The agent will:

* Use LLM to **think through** the query
* Choose tools based on descriptions
* Chain tool outputs + reasoning

---

## 🧠 Advanced Use Case: Multi-Modal Agent

You can register tools that:

* Call **APIs** (weather, news, calendar)
* **Run Python code** for plotting, analysis
* Chat over **PDFs**, **Notion**, **emails**, etc.

---

## 🧩 Custom Tool from Class (Optional)

You can also use a tool as a **callable class**:

```python
class JokeTool:
    name = "JokeGenerator"
    description = "Generates a random joke."

    def __call__(self, query: str) -> str:
        return "Why did the AI cross the road? To optimize both sides."

joke_tool = Tool.from_function(JokeTool())
```

---

## 📦 Best Practices for Custom Tools

| Practice                         | Reason                                                   |
| -------------------------------- | -------------------------------------------------------- |
| Use clear `name` + `description` | Helps LLM choose the correct tool                        |
| Keep input/output simple         | Most tools should work with string I/O                   |
| Handle errors gracefully         | Prevent agent from crashing                              |
| Make tools atomic                | One task per tool (e.g. "get\_weather", not "get\_info") |
| Limit external API calls         | Prevent long latency and rate limits                     |

---

## ✅ Summary

| Element       | Purpose                              |
| ------------- | ------------------------------------ |
| `Tool`        | Wraps a callable Python function     |
| `Agent`       | Decides which tool to use and when   |
| `LLM`         | Drives reasoning and decision-making |
| `VectorStore` | Enables knowledge retrieval (RAG)    |

---

---

## 🪲 **18\_debugging\_langchain\_apps\_logging\_tracing.md**

> 📌 *Topic: Debugging LangChain apps and integrating proper logging and tracing*

---

## 🔍 Why Debugging and Logging Matter

LangChain apps involve:

* 🧠 LLMs with unpredictable outputs
* 🔗 Chains that route inputs across components
* 🔧 Tools and agents that reason and take actions
* 🧠 Memory that stores state

Proper debugging:

* Speeds up development
* Catches broken tools or prompts
* Helps visualize and **optimize multi-step reasoning**

---

## 🔧 Built-in LangChain Logging

LangChain offers built-in logging with:

* `verbose=True` flag
* Environment variables
* Tracing integrations (like LangSmith)

---

### ✅ Option 1: Use `verbose=True`

Pass `verbose=True` when initializing chains or agents:

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

This logs:

* Prompts
* Thoughts and Actions
* Tool inputs/outputs
* Final answers

---

### ✅ Option 2: Use `LANGCHAIN_VERBOSE=1`

Enable global verbose mode with an environment variable:

```bash
export LANGCHAIN_VERBOSE=1
```

---

## 📦 Enable Advanced Debugging with LangSmith (Recommended)

**[LangSmith](https://smith.langchain.com/)** is LangChain’s official observability platform for:

* Tracing prompt flow
* Visualizing chains
* Sharing debugging sessions

### 🔐 Step 1: Sign up and get API Key

Get your API key from:
👉 [https://smith.langchain.com/](https://smith.langchain.com/)

---

### 🛠️ Step 2: Set LangSmith ENV variables

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your_langsmith_api_key"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

---

### 🔍 Step 3: Enable tracing in code

```python
from langchain.callbacks.tracers import LangChainTracer
from langchain.agents import initialize_agent

tracer = LangChainTracer(project_name="DebugDemo")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    callbacks=[tracer]
)
```

You’ll now see a **full visual trace** in your LangSmith dashboard.

---

## 📜 Logging with Python’s `logging` Module

You can also plug in Python’s logging system:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting the chatbot...")

response = agent.run("What is 25 * 12?")
logger.info(f"Response: {response}")
```

---

## 🧪 Add Callback Handlers (Stream Output / Logging / Custom)

### Example: Print token streaming output live

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

---

## 🧩 Custom Debug Callback

Create a custom callback for **event-based logging**:

```python
from langchain.callbacks.base import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("[DEBUG] Chain started:", inputs)

    def on_chain_end(self, outputs, **kwargs):
        print("[DEBUG] Chain ended:", outputs)

    def on_tool_start(self, serialized, input_str, **kwargs):
        print("[DEBUG] Tool called with:", input_str)

    def on_tool_end(self, output, **kwargs):
        print("[DEBUG] Tool output:", output)
```

Add it like this:

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[DebugCallback()],
    verbose=True
)
```

---

## 📊 Monitor Memory and State

If you’re using `ConversationBufferMemory`, you can inspect it:

```python
print(memory.buffer_as_messages)
```

---

## ✅ Summary

| Tool/Method              | Use Case                              |
| ------------------------ | ------------------------------------- |
| `verbose=True`           | See internal LLM/Agent steps          |
| `LANGCHAIN_VERBOSE=1`    | Global verbose mode                   |
| `LangSmith`              | Full visual trace of execution        |
| `StreamingStdOutHandler` | Real-time streaming output            |
| Python `logging`         | Custom logs and trace files           |
| `CustomCallbackHandler`  | Track tool/chain/LLM actions yourself |

---

---

## 🧱 Pipeline Architecture

```
Documents (PDFs, etc.)
      ↓
Text Splitter
      ↓
Embeddings (OpenAI/Cohere/HuggingFace)
      ↓
Pinecone Vector Store
      ↓
Query
      ↓
Top-k Matching Documents
```

---

## 📦 Step 1: Install Dependencies

```bash
pip install langchain pinecone-client openai tiktoken pdfminer.six
```

---

## 🔐 Step 2: Set Up API Keys

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_env (e.g. us-east4-gcp)
```

Load them in Python:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

---

## 📚 Step 3: Load and Chunk Your Document

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("example.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
```

---

## 🧬 Step 4: Generate Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
```

---

## 🌲 Step 5: Initialize Pinecone and Upload Data

```python
import pinecone

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = "langchain-vector-search"

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

from langchain.vectorstores import Pinecone

# Create the Pinecone vector store
vectorstore = Pinecone.from_documents(docs, embedding, index_name=index_name)
```

✅ This step embeds your text and pushes the vectors to Pinecone.

---

## 🔍 Step 6: Perform a Semantic Query

```python
query = "What is LangChain and how does it help developers?"

# Get top 3 most similar chunks
results = vectorstore.similarity_search(query, k=3)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:\n{result.page_content}\n")
```

---

## 🧠 Bonus: Use It with a RAG Chain

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OpenAI()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

response = qa_chain.run(query)
print("\nAnswer:", response)
```

---

## ✅ Summary

| Step             | LangChain Component               |
| ---------------- | --------------------------------- |
| Load documents   | `PyPDFLoader`, `TextLoader`       |
| Split text       | `RecursiveCharacterTextSplitter`  |
| Embed documents  | `OpenAIEmbeddings`                |
| Store vectors    | `Pinecone.from_documents`         |
| Query vectors    | `.similarity_search()`            |
| Retrieve for RAG | `.as_retriever()` + `RetrievalQA` |

---

---

## 🧩 Why This Matters

LangChain apps often involve:

* Complex **prompt engineering**
* **Chain composition** for multi-step logic
* Managing **LLM behavior** via parameters like `temperature`

Small tweaks here = big improvements in:

* Response quality ✅
* Factual accuracy 📊
* Creative generation 🎨

---

## 🎯 Goals

1. **Design effective prompts** using `PromptTemplate`
2. **Choose the right chain type** (`stuff`, `map_reduce`, `refine`)
3. **Tune temperature** for desired creativity/consistency

---

## 1️⃣ Tune PromptTemplates 🧠

### 💡 Use `PromptTemplate` with input variables

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} like I’m five."
)

print(prompt.format(topic="blockchain"))
```

---

### 🔁 Chain prompt + LLM

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.5)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("quantum computing"))
```

---

### 🛠 Prompt Engineering Tips

| Tip                      | Example                              |
| ------------------------ | ------------------------------------ |
| Ask for format           | "List 3 bullet points about {topic}" |
| Give persona             | "You are a history teacher..."       |
| Add constraints          | "Use no more than 30 words."         |
| Show examples (few-shot) | Add 2–3 Q\&A pairs in prompt         |

---

## 2️⃣ Select the Best Chain Type 🔗

### 🔹 Stuff Chain (default)

Concatenates all docs into a single prompt.

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

> ✅ Fast
> ❌ Context window limited (\~8K-32K tokens)

---

### 🔹 Map Reduce Chain

* **Map**: LLM processes each chunk separately
* **Reduce**: LLM summarizes or combines

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce"
)
```

> ✅ Good for long docs
> ❌ Slower (multi-round LLM calls)

---

### 🔹 Refine Chain

* Process one doc
* Refine the answer with each additional doc

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine"
)
```

> ✅ Improves accuracy over steps
> ❌ Can be repetitive if not tuned

---

## 3️⃣ Tune LLM Temperature 🔥

| Temperature | Behavior                     |
| ----------- | ---------------------------- |
| `0.0`       | Deterministic, fact-focused  |
| `0.3–0.5`   | Balanced, business responses |
| `0.7–1.0`   | Creative, marketing, stories |

```python
llm = OpenAI(temperature=0.2)  # factual
llm = OpenAI(temperature=0.8)  # creative
```

Use **low temperature** for:

* RAG
* QA bots
* Code generation

Use **high temperature** for:

* Idea generation
* Dialogue
* Storytelling

---

## 🧪 Example: Same Prompt, Different Temps

```python
prompt = PromptTemplate.from_template("Write a tagline for a tech company.")

llm1 = OpenAI(temperature=0.1)
llm2 = OpenAI(temperature=0.9)

print("Low Temp:", LLMChain(llm=llm1, prompt=prompt).run({}))
print("High Temp:", LLMChain(llm=llm2, prompt=prompt).run({}))
```

---

## 🎯 Summary Table

| Component   | What to Tune                    | Impact                               |
| ----------- | ------------------------------- | ------------------------------------ |
| Prompt      | Tone, structure                 | Better outputs, fewer hallucinations |
| Chain Type  | `stuff`, `refine`, `map_reduce` | Tradeoff between speed & accuracy    |
| Temperature | `0.0` to `1.0`                  | Control randomness/creativity        |

---

## ✅ Best Practices

* ✅ Use **low temp** + **structured prompt** for precision (QA, RAG)
* ✅ Use **map\_reduce** or **refine** for large or legal/scientific texts
* ✅ Use `PromptTemplate` + `FewShotPromptTemplate` to guide tone and format
* ✅ Evaluate performance across chain types using LangSmith traces

---

---

## ✅ What You'll Build

An app that:

* 🧠 Understands context with memory
* 📄 Retrieves answers from a custom knowledge base
* 🔍 Uses **FAISS** for fast vector search
* 🤖 Answers user questions with OpenAI's LLM
* 🧩 Modular, reusable, and easy to scale

---

## 📦 Dependencies

```bash
pip install langchain openai faiss-cpu pdfminer.six tiktoken
```

Optional (for web apps):

```bash
pip install streamlit
```

---

## 🔐 Setup `.env` (for API keys)

```env
OPENAI_API_KEY=your_openai_key
```

---

## 📁 Project Structure

```
📁 context_aware_qa/
├── app.py                  # Main runner
├── knowledge_base.pdf      # Your source document
├── .env
```

---

## 🧱 Step 1: Load and Split Document

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("knowledge_base.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
```

---

## 🧬 Step 2: Create Embeddings and Store in FAISS

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embedding)
```

You can persist the index like this (optional):

```python
vectorstore.save_local("faiss_index")
# Later load:
# vectorstore = FAISS.load_local("faiss_index", embedding)
```

---

## 🔎 Step 3: Create Retriever

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

---

## 🧠 Step 4: Setup LLM + Memory

```python
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

---

## 🔗 Step 5: Create the RetrievalQA Chain

```python
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)
```

---

## 💬 Step 6: Interactive Chat Loop

```python
print("📘 Context-Aware Assistant Ready! Type 'exit' to quit.\n")

chat_history = []

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run({"question": query, "chat_history": chat_history})
    print("Bot:", response)
    chat_history.append((query, response))
```

---

## 🧪 Example Chat

```
You: What is LangChain?
Bot: LangChain is a framework for building applications powered by large language models...

You: Who created it?
Bot: LangChain was created by Harrison Chase.
```

🧠 It **remembers previous questions** and **uses FAISS** to give document-grounded answers.

---

## 🧰 Optional: Streamlit UI

```python
# app.py
import streamlit as st

st.title("💬 Context-Aware Q&A Assistant")

user_input = st.text_input("Ask a question:")

if user_input:
    response = qa_chain.run(user_input)
    st.write(response)
```

Run with:

```bash
streamlit run app.py
```

---

## ✅ Summary

| Component       | Tool Used                        |
| --------------- | -------------------------------- |
| Document Loader | `PyPDFLoader`                    |
| Text Splitter   | `RecursiveCharacterTextSplitter` |
| Embedding       | `OpenAIEmbeddings`               |
| Vector Store    | `FAISS`                          |
| Retriever       | `.as_retriever()`                |
| Memory          | `ConversationBufferMemory`       |
| Chain           | `ConversationalRetrievalChain`   |

---

## 🧠 Bonus Tips

* Use `ConversationalSummaryMemory` for long sessions
* Replace `OpenAI` with `HuggingFaceHub` or `Ollama` for local use
* Combine with LangSmith to trace performance

---

## 📌 Final Checklist

✅ Document-based context
✅ Memory-aware conversation
✅ Vector search via FAISS
✅ Modular code for scaling
✅ Compatible with Streamlit UI

---
