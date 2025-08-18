
---

## 🧩 Why This Matters

LangChain apps often involve:

* Complex **prompt engineering**
* **Chain composition** for multi-step logic
* Managing **LLM behavior** via parameters like `temperature`

Small tweaks here = big improvements in:

* Response quality ✅
* Factual accuracy 📊
* Creative generation 🎨

---

## 🎯 Goals

1. **Design effective prompts** using `PromptTemplate`
2. **Choose the right chain type** (`stuff`, `map_reduce`, `refine`)
3. **Tune temperature** for desired creativity/consistency

---

## 1️⃣ Tune PromptTemplates 🧠

### 💡 Use `PromptTemplate` with input variables

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} like I’m five."
)

print(prompt.format(topic="blockchain"))
```

---

### 🔁 Chain prompt + LLM

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.5)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("quantum computing"))
```

---

### 🛠 Prompt Engineering Tips

| Tip                      | Example                              |
| ------------------------ | ------------------------------------ |
| Ask for format           | "List 3 bullet points about {topic}" |
| Give persona             | "You are a history teacher..."       |
| Add constraints          | "Use no more than 30 words."         |
| Show examples (few-shot) | Add 2–3 Q\&A pairs in prompt         |

---

## 2️⃣ Select the Best Chain Type 🔗

### 🔹 Stuff Chain (default)

Concatenates all docs into a single prompt.

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

> ✅ Fast
> ❌ Context window limited (\~8K-32K tokens)

---

### 🔹 Map Reduce Chain

* **Map**: LLM processes each chunk separately
* **Reduce**: LLM summarizes or combines

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce"
)
```

> ✅ Good for long docs
> ❌ Slower (multi-round LLM calls)

---

### 🔹 Refine Chain

* Process one doc
* Refine the answer with each additional doc

```python
RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine"
)
```

> ✅ Improves accuracy over steps
> ❌ Can be repetitive if not tuned

---

## 3️⃣ Tune LLM Temperature 🔥

| Temperature | Behavior                     |
| ----------- | ---------------------------- |
| `0.0`       | Deterministic, fact-focused  |
| `0.3–0.5`   | Balanced, business responses |
| `0.7–1.0`   | Creative, marketing, stories |

```python
llm = OpenAI(temperature=0.2)  # factual
llm = OpenAI(temperature=0.8)  # creative
```

Use **low temperature** for:

* RAG
* QA bots
* Code generation

Use **high temperature** for:

* Idea generation
* Dialogue
* Storytelling

---

## 🧪 Example: Same Prompt, Different Temps

```python
prompt = PromptTemplate.from_template("Write a tagline for a tech company.")

llm1 = OpenAI(temperature=0.1)
llm2 = OpenAI(temperature=0.9)

print("Low Temp:", LLMChain(llm=llm1, prompt=prompt).run({}))
print("High Temp:", LLMChain(llm=llm2, prompt=prompt).run({}))
```

---

## 🎯 Summary Table

| Component   | What to Tune                    | Impact                               |
| ----------- | ------------------------------- | ------------------------------------ |
| Prompt      | Tone, structure                 | Better outputs, fewer hallucinations |
| Chain Type  | `stuff`, `refine`, `map_reduce` | Tradeoff between speed & accuracy    |
| Temperature | `0.0` to `1.0`                  | Control randomness/creativity        |

---

## ✅ Best Practices

* ✅ Use **low temp** + **structured prompt** for precision (QA, RAG)
* ✅ Use **map\_reduce** or **refine** for large or legal/scientific texts
* ✅ Use `PromptTemplate` + `FewShotPromptTemplate` to guide tone and format
* ✅ Evaluate performance across chain types using LangSmith traces

---

## 🚀 Next Steps

Would you like a:

* ⚙️ Prompt tuning dashboard in Streamlit?
* 📄 Comparison script to evaluate chain types?
* 🧠 RAG + temperature tuning CLI tool?

