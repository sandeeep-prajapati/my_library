
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

Would you like a version that:

* Accepts PDF uploads for **chat with documents**?
* Uses **Chroma + RAG + OpenAI** together?
* Works **offline using Ollama LLMs**?

