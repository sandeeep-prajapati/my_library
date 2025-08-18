
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

Would you like:

* A downloadable `.py` script?
* A Dockerized version?
* A multi-document search version?

