# 📄 Document Intelligence System (Offline RAG + Analytics + Knowledge Graph)

An AI-powered **Document Intelligence System** that allows users to upload PDFs and:
- 💬 Ask questions (RAG-based Q&A)
- 📊 Analyze document topics and keywords
- 🔗 Explore relationships via a Knowledge Graph

This system runs **fully offline** using a local LLM (Mistral 7B GGUF) — no API required.

---

## 🚀 Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Local LLM inference (Mistral via llama.cpp)
- ✅ Context-aware question answering
- ✅ Topic clustering (KMeans)
- ✅ Keyword extraction
- ✅ Knowledge Graph generation
- ✅ Streamlit UI

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- sentence-transformers
- FAISS
- llama-cpp-python (GGUF models)
- spaCy
- scikit-learn
- networkx + matplotlib

---

## 📁 Project Structure

```
project_root/
│
├── app/
│   └── app.py              # Streamlit UI
│
├── core/
│   ├── ingestion.py
│   ├── embedding.py
│   ├── vectorstore.py
│   ├── rag.py
│   ├── compression.py
│   ├── analytics.py
│   ├── graph.py
│   ├── llm.py
│
├── data/                   # Uploaded PDFs
├── models/                 # GGUF model goes here
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

---

### 2️⃣ Create Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv env
env\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv env
source env/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

### 5️⃣ Download LLM Model (IMPORTANT)

You must manually download a GGUF model.

### Recommended:

* **Mistral 7B Instruct (GGUF, Q4_K_M)**

👉 Example:

```
mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

### 📁 Place the model here:

```
models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

### ⚠️ Important

* Do NOT rename the file unless you update `core/llm.py`
* Ensure enough RAM (~6GB+) is available

---

## ▶️ Running the App

```bash
streamlit run app/app.py
```

---

## 🧠 How It Works

### 1. Document Processing

* PDF → Text extraction (PyMuPDF)
* Text → Chunking

### 2. Embedding + Retrieval

* sentence-transformers embeddings
* FAISS vector search

### 3. RAG Pipeline

* Query → retrieve relevant chunks
* Context compression
* LLM generates answer

### 4. Analytics

* KMeans clustering for topics
* Keyword frequency extraction

### 5. Knowledge Graph

* Entity extraction (spaCy)
* Graph construction (networkx)

---

## 📊 Example Outputs

### 💬 Q&A

> "What is the main idea of the paper?"

✔ Structured multi-sentence answer using context

---

### 📊 Analytics

* Topic clusters
* Top keywords (financial, transformer, hawkes, etc.)

---

### 🔗 Knowledge Graph

* Entities connected by relationships
* Visual graph saved as `graph.png`

---

## ⚠️ Notes

* First run may be slower due to model loading
* CPU inference is supported (GPU optional)
* Context length is limited to avoid LLM crashes

---

## 🚀 Future Improvements

* Multi-document support
* Better entity relationship extraction
* Streaming responses
* GPU acceleration optimization
* Advanced UI enhancements

---

## 👨‍💻 Author

Built as part of an AI Systems / Document Intelligence project.

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
