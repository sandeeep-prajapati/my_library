
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

Would you like a **LangChain Agent Playground app** with:

* Custom tool integration
* Agent type selection
* Memory toggle
* Chat UI (Streamlit/Gradio)?
