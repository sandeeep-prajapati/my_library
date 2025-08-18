
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

Let me know if you’d like:

* A starter project combining chains + memory
* A comparison chart of memory performance
* A Streamlit chatbot using memory types
