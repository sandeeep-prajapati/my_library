
---

## 📄 **11\_loading\_data\_from\_pdfs\_notion\_csv\_webpages.md**

> 📌 *Topic: Load data from PDFs, Notion, CSV, and web pages using built-in loaders*

---

## 📦 Why Use Loaders?

LangChain provides **`DocumentLoader`** classes to easily ingest data from various formats into a unified `Document` structure.

Each loader returns a list of `Document` objects like:

```python
Document(
    page_content="text",
    metadata={"source": "file name", "page": 3}
)
```

---

## ⚙️ 1. Installing Loaders

```bash
pip install langchain unstructured pdfminer.six beautifulsoup4 requests python-docx html2text
```

> Also required: API tokens for Notion, if applicable.

---

## 📚 2. Load PDF Files

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("files/myfile.pdf")
docs = loader.load()

print(docs[0].page_content[:300])  # Preview content
```

### 🧠 Other PDF loaders:

* `PDFMinerLoader`
* `PDFPlumberLoader`
* `PyMuPDFLoader`

---

## 🗂️ 3. Load CSV Files

Each **row** becomes a `Document`.

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader(file_path="files/data.csv")
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
```

✅ Supports large CSVs. You can customize field parsing with `encoding`, `delimiter`, etc.

---

## 📝 4. Load Notion Pages (via API)

### Step 1: Install required dependencies

```bash
pip install notion-client
```

### Step 2: Load documents from Notion

```python
from langchain.document_loaders import NotionDBLoader

loader = NotionDBLoader(
    integration_token="your_notion_api_key",
    database_id="your_notion_db_id"
)

docs = loader.load()
```

> 📌 Get your [Notion API Key & DB ID](https://developers.notion.com/docs/getting-started)

---

## 🌐 5. Load Web Pages (HTML URLs)

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")
docs = loader.load()

print(docs[0].page_content[:300])
```

> Uses `requests` + `BeautifulSoup` internally
> ✅ Great for scraping blogs, docs, wiki pages

---

## 🖼️ 6. Load from Markdown or Text

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("files/notes.txt", encoding='utf-8')
docs = loader.load()
```

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("files/notes.md")
docs = loader.load()
```

---

## 📸 7. Load Images (OCR with Tesseract)

```bash
pip install pytesseract pillow
```

```python
from langchain.document_loaders import UnstructuredImageLoader

loader = UnstructuredImageLoader("images/screenshot.png")
docs = loader.load()
```

> Useful for **scanned PDFs**, **handwritten notes**, or **screenshots**

---

## 🧠 Bonus: Combine Multiple Sources

```python
from langchain.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader

pdf_docs = PyPDFLoader("sample.pdf").load()
csv_docs = CSVLoader(file_path="data.csv").load()
web_docs = WebBaseLoader("https://example.com").load()

all_docs = pdf_docs + csv_docs + web_docs
```

---

## 🧩 Next Step: Split Documents into Chunks

Once loaded, split large documents into chunks for processing:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs_chunked = splitter.split_documents(all_docs)
```

---

## ✅ Summary Table

| Source        | Loader Class                    | Notes                      |
| ------------- | ------------------------------- | -------------------------- |
| **PDF**       | `PyPDFLoader`, `PDFMinerLoader` | Extracts text page-wise    |
| **CSV**       | `CSVLoader`                     | One row = one document     |
| **Notion**    | `NotionDBLoader`                | Needs API token + DB ID    |
| **Web pages** | `WebBaseLoader`                 | HTML to text using `bs4`   |
| **Images**    | `UnstructuredImageLoader`       | OCR-based text from images |
| **Text/MD**   | `TextLoader`, `MarkdownLoader`  | Loads raw text             |

---

Would you like a sample **data ingestion pipeline** that:

* Loads PDF/CSV/Web
* Splits into chunks
* Embeds into FAISS or Chroma
* Supports retrieval with LangChain?
