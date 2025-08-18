
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

## 🧪 Mini Project Idea

Build a RAG chatbot that:

* Loads PDFs and Notion notes
* Chunks and embeds with `OpenAIEmbeddings`
* Stores in **Chroma** or **FAISS**
* Queries and returns context-aware answers

