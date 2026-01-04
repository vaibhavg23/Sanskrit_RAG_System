# 🕉️ Sanskrit Document Retrieval-Augmented Generation (RAG) System

An AI-powered **Sanskrit Document Retrieval-Augmented Generation (RAG) system** built as part of an **AI/ML Intern Assignment**.  
The system retrieves relevant Sanskrit documents using vector similarity search and optionally generates contextual answers using a lightweight language model — all running **entirely on CPU**.

---

## 📌 Project Objectives

- Enable efficient retrieval of Sanskrit text documents
- Support both **Sanskrit (Devanagari)** and **English** queries
- Implement a **CPU-only** Retrieval-Augmented Generation pipeline
- Maintain a **modular, explainable, and reproducible architecture**

---

## 🧠 System Architecture (High Level)

User Query
↓
Text Embedding (Sentence Transformers)
↓
FAISS Vector Search
↓
Top-K Relevant Document Chunks
↓
(Optional) LLM-based Answer Generation
↓
Final Response



---

## 📂 Project Structure


immverse AI/
│
├── src/
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── document_loader.py
│   ├── llm_generator.py
│   ├── logger.py
│   ├── config.py
│   └── __init__.py
│
├── data/
│   └── *.txt
│
├── report/
│   └── Sanskrit_RAG_System_Report.pdf
│
├── assets/
│   └── Sanskrit_RAG_Architecture.png   
│
├── requirements.txt
├── README.md
└── venv/   (optional / not submitted)



---

## 📊 Dataset Description

- **Format:** Plain text (`.txt`)
- **Language:** Sanskrit (Devanagari) and English
- **Content:** Classical Sanskrit stories, prose passages, and moral narratives
- **Storage:** Local filesystem (offline, no external API dependency)
- **Encoding:** UTF-8

---

## ⚙️ Technologies Used

- **Python 3.9+**
- **Streamlit** – Web interface
- **Sentence Transformers** – Text embeddings
- **FAISS** – Vector similarity search
- **Hugging Face Transformers** – Optional language model generation
- **CPU-only inference** (no GPU required)

---

## 🚀 How to Run the Project

### 1️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows


pip install -r requirements.txt

streamlit run src/app.py

🔍 Example Queries

मूर्खभृत्यस्य कथा किम्?

कालीदासस्य चातुर्यं वर्णयतु

प्रयत्नस्य महत्त्वं किम्?

What is the story of the foolish servant?

Tell me about Kalidasa's cleverness

📈 Performance Observations

Latency: Sub-second retrieval for small document collections

Accuracy: Relevant document chunks retrieved for factual and story-based queries

Resource Usage: CPU-only, suitable for low-resource environments

⚠️ Limitations

Optimized for small to medium document collections

LLM-based generation on CPU may increase response time

Sanskrit semantic understanding depends on embedding quality

🔮 Future Enhancements

Sanskrit-specific embedding models

Improved summarization of retrieved content

Larger corpus support

Advanced evaluation metrics