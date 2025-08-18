
---

## 🤝 **09\_integrating\_llms\_openai\_huggingface\_cohere.md**

> 📌 *Topic: Integrate OpenAI, Hugging Face, Cohere, and other LLMs in LangChain*

---

## 🎯 Why Use Multiple LLM Providers?

| Reason                  | Benefit                                       |
| ----------------------- | --------------------------------------------- |
| 💵 Cost control         | Choose cheaper models for less critical tasks |
| ⚡ Performance tuning    | Pick faster models where latency matters      |
| 🧠 Model specialization | Some models perform better on certain domains |
| 💬 Fallbacks & backups  | Switch if one provider is unavailable         |
| 🔌 Hybrid architectures | Use multiple models in a single workflow      |

---

## 📦 1. Install Required Packages

```bash
pip install langchain openai cohere transformers huggingface_hub
```

---

## 🔑 2. Set API Keys

You can either export environment variables:

```bash
export OPENAI_API_KEY=your-openai-key
export COHERE_API_KEY=your-cohere-key
export HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
```

Or use a `.env` file with `python-dotenv`.

---

## 🧠 3. OpenAI Integration

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
response = llm("What is LangChain?")
print(response)
```

You can configure:

* `model_name`: e.g., `"gpt-3.5-turbo"`, `"gpt-4"`
* `max_tokens`, `temperature`, etc.

---

## 🤗 4. Hugging Face Integration (via `HuggingFaceHub`)

```python
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 100}
)

print(llm("Explain neural networks in simple terms"))
```

> Hugging Face supports both **transformer models** (like T5, GPT2) and **inference API** (paid tier).

---

## 🧬 5. Hugging Face Local Model (via `HuggingFacePipeline`)

If you want to **run models locally**, use:

```python
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

print(llm("Write a short poem about a sunrise."))
```

> ⚠️ Local models require more RAM/GPU.

---

## 🧪 6. Cohere Integration

```python
from langchain.llms import Cohere

llm = Cohere(
    model="command-nightly",
    temperature=0.5,
    max_tokens=100
)

print(llm("Summarize the history of India."))
```

> Cohere is fast and focused on business-ready LLMs.

---

## 🔄 7. Switching LLMs in a Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Translate to French: {sentence}")
sentence = "I love AI."

# Using OpenAI
openai_chain = LLMChain(llm=OpenAI(), prompt=prompt)
print(openai_chain.run(sentence=sentence))

# Using Cohere
cohere_chain = LLMChain(llm=Cohere(), prompt=prompt)
print(cohere_chain.run(sentence=sentence))
```

> You can switch `llm=` dynamically in your pipeline!

---

## ⚙️ 8. Building a Multi-Model Agent

You can even assign different tasks to different models.

```python
summarizer = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo"),
    prompt=PromptTemplate.from_template("Summarize: {text}")
)

translator = LLMChain(
    llm=HuggingFaceHub(repo_id="t5-base"),
    prompt=PromptTemplate.from_template("Translate to German: {text}")
)
```

---

## 📊 LLM Provider Comparison

| Provider         | Pros                         | Cons                           |
| ---------------- | ---------------------------- | ------------------------------ |
| **OpenAI**       | Top-tier quality, chat modes | Cost, rate limits              |
| **Hugging Face** | Flexible, many models        | May need paid API or local GPU |
| **Cohere**       | Fast, cheaper                | Fewer model types              |
| **Local (HF)**   | Full control, privacy        | Resource intensive             |

---

## ✅ Summary

| LLM Type          | Class in LangChain      |
| ----------------- | ----------------------- |
| OpenAI            | `langchain.llms.OpenAI` |
| HuggingFace Hub   | `HuggingFaceHub`        |
| HuggingFace Local | `HuggingFacePipeline`   |
| Cohere            | `Cohere`                |

---

## 🚀 Next Steps

* Use different LLMs in **different chains** (e.g., summarization via OpenAI, sentiment via Cohere)
* Add **fallbacks** if one provider fails
* Tune **temperature, max tokens, and prompts** per provider

---
