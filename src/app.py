"""
Streamlit Web Interface for Sanskrit RAG System
"""

import streamlit as st
import sys
import os

# ✅ ADD PROJECT ROOT TO PYTHON PATH (CRITICAL FIX)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_pipeline import SanskritRAGPipeline
from src.config import TOP_K_DOCS

# Page config
st.set_page_config(
    page_title="Sanskrit RAG System",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False

# ✅ UPDATED CSS (TEXT VISIBLE IN BOTH LIGHT & DARK MODE)
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.8rem;
        color: #EAEAEA;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* 🔥 DOCUMENT BOX – FORCE DARK TEXT */
    .doc-box {
        background: #FFF3E0;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 6px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);

        color: #1A1A1A !important;
        font-size: 1.1rem;
        line-height: 1.7;
    }

    .doc-box * {
        color: #1A1A1A !important;
    }

    /* 🔥 RESPONSE BOX – FORCE DARK TEXT */
    .response-box {
        background: #E3F2FD;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 6px solid #2E86AB;
        font-size: 1.3rem;
        line-height: 2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);

        color: #102027 !important;
    }

    .response-box * {
        color: #102027 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to set query
def set_query(query_text):
    st.session_state.current_query = query_text
    st.session_state.trigger_search = True

# Load RAG pipeline (cached)
@st.cache_resource
def load_rag_pipeline():
    return SanskritRAGPipeline()

# Header
st.markdown('<div class="main-header">🕉️ Sanskrit RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">संस्कृत दस्तावेज पुनर्प्राप्ति-संवर्धित उत्पादन प्रणाली</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    k_docs = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=5,
        value=TOP_K_DOCS
    )

    use_llm = st.checkbox(
        "Use LLM Generation",
        value=False
    )

    st.markdown("## 📖 Sample Queries")

    sample_queries = [
        "मूर्खभृत्यस्य कथा किम्?",
        "कालीदासस्य चातुर्यं वर्णयतु",
        "वृद्धायाः कथायां किं घटितम्?",
        "प्रयत्नस्य महत्त्वं किम्?",
        "What is the story of the foolish servant?",
        "Tell me about Kalidasa's cleverness"
    ]

    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            set_query(sq)
            st.rerun()

# Main input
query = st.text_input(
    "Query",
    value=st.session_state.current_query,
    placeholder="Enter your question in Sanskrit or English...",
    label_visibility="collapsed"
)

search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Process query
if (search_button or st.session_state.trigger_search) and query:
    st.session_state.trigger_search = False

    with st.spinner("🔄 Processing your query..."):
        try:
            rag = load_rag_pipeline()
            result = rag.query(query, k=k_docs, use_llm=use_llm)

            if 'error' in result:
                st.error(result['error'])
            else:
                st.success(f"✅ Query processed in {result['latency']:.2f} seconds")

                with st.expander("📄 Retrieved Documents", expanded=True):
                    for i, doc in enumerate(result['retrieved_docs']):
                        st.markdown(f"""
                        <div class="doc-box">
                            <strong>📄 Document {i+1}</strong><br>
                            <em>{doc['metadata'].get('title','')}</em><br>
                            <b>Score:</b> {doc['similarity_score']:.4f}<br><br>
                            {doc['content'][:400]}...
                        </div>
                        """, unsafe_allow_html=True)

                answer_text = result.get("answer", "No answer generated.")
                formatted_answer = answer_text.replace("\n", "<br>")

                st.markdown(f"""
                <div class="response-box">
                    {formatted_answer}
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("⏱️ Latency", f"{result['latency']:.2f}s")
                col2.metric("📚 Docs Retrieved", result['num_docs_retrieved'])
                col3.metric("🤖 LLM Used", "Yes" if use_llm else "No")

        except Exception as e:
            st.error(str(e))
            st.exception(e)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem; color:#B0BEC5;">
    🕉️ <b>Sanskrit RAG System</b><br>
    FAISS • Sentence Transformers • Streamlit
</div>
""", unsafe_allow_html=True)
