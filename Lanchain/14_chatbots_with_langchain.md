
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

## 💡 Next Steps

Would you like me to generate a:

* ✅ Streamlit interface for this chatbot?
* ✅ Version with **file upload + RAG tools**?
* ✅ Version with **tool fallback + error handling**?
* ✅ Version using **local LLM (Ollama)**?

