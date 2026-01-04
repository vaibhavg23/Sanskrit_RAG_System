# 📊 Evaluation & Performance Notes
Sanskrit Document Retrieval-Augmented Generation (RAG) System

This document records the qualitative and quantitative evaluation observations
for the Sanskrit RAG system developed as part of the AI/ML Intern Assignment.
These notes support the evaluation and performance sections of the final report.

---

## 1. Evaluation Environment

- **Execution Mode:** CPU-only
- **GPU Usage:** None
- **Operating System:** Local machine (Windows/Linux)
- **Python Version:** 3.x
- **Frameworks Used:** Streamlit, FAISS, Sentence Transformers, Hugging Face
- **Deployment:** Local Streamlit application

The system was intentionally designed to operate without GPU acceleration
to comply with low-resource and assignment constraints.

---

## 2. Dataset Overview

- **Format:** Plain text (`.txt`)
- **Language:** Sanskrit (Devanagari) and English
- **Content Type:** Classical Sanskrit stories, prose passages, moral narratives
- **Encoding:** UTF-8
- **Corpus Size:** Small to medium document collection

---

## 3. Test Queries Used

### Sanskrit Queries
- मूर्खभृत्यस्य कथा किम्?
- प्रयत्नस्य महत्त्वं किम्?
- वृद्धायाः कथायां किं घटितम्?

### English Queries
- What is the story of the foolish servant?
- Tell me about Kalidasa's cleverness
- What is the moral of the story?

---

## 4. Performance Metrics (Observed)

### 4.1 Latency

| Operation Type | Approximate Time |
|---------------|------------------|
| Document Retrieval (FAISS) | 0.3 – 0.6 seconds |
| Retrieval + LLM Generation | 1.5 – 3.0 seconds |

Latency values were observed directly from the Streamlit interface.

---

### 4.2 Memory Usage

- Moderate memory usage due to in-memory embeddings
- Suitable for small to medium document collections
- No excessive memory spikes observed

---

## 5. Retrieval Accuracy (Qualitative)

- Retrieved document chunks were generally **relevant to the query**
- Story-based and factual queries performed well
- Sanskrit keyword matching and semantic similarity were effective
- Occasional partial context retrieval observed for abstract queries

No automated evaluation metrics (BLEU/ROUGE) were used, as the assignment
focuses on qualitative assessment.

---

## 6. Generation Quality (LLM Enabled)

- Generated responses were:
  - Contextually aligned with retrieved documents
  - Grammatically coherent in English
  - Limited but usable for Sanskrit-based queries
- LLM generation improved readability but increased response time

---

## 7. Observations

- FAISS enabled fast and efficient similarity search
- Sentence Transformer embeddings handled multilingual queries reasonably well
- CPU-only LLM inference introduced noticeable latency for longer responses
- System performed best on factual and story-related queries

---

## 8. Limitations Observed

- Sanskrit semantic understanding depends heavily on embedding quality
- Large document collections may degrade retrieval speed
- LLM-based generation is slower on CPU
- No advanced quantitative evaluation metrics implemented

---

## 9. Conclusion

The Sanskrit RAG system demonstrates effective document retrieval and
context-aware response generation under strict CPU-only constraints.
Performance is satisfactory for intern-level expectations and small-scale
datasets, with clear scope for future optimization.

---

## 10. Notes

These evaluation notes are intended to supplement the technical report
and provide transparency into system behavior during testing.
