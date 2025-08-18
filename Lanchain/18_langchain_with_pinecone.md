
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

## 📁 Want a Full File?

Would you like me to generate:

* A single `.py` file that runs the whole pipeline?
* A Streamlit app version?
* A variant using **local LLMs** or **Chroma** instead of Pinecone?

