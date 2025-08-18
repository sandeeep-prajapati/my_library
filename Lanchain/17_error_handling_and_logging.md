
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

## 🚀 Next Steps

Would you like me to:

* Build a **LangSmith-integrated Streamlit RAG bot**?
* Show **debugging examples on PDF tools**?
* Help you trace **broken prompts in agents**?

