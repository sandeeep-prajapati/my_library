
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

## ⚙️ Next Steps

Would you like me to generate:

* ✅ A **PDF/CSV QA tool** to plug into your agent?
* ✅ A **financial assistant** with APIs + memory?
* ✅ A **modular Streamlit chatbot** using these tools?
