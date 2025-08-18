
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
