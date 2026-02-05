# ---------------- IMPORTS ----------------
import streamlit as st
from rag_pipeline import hybrid_rag_answer, evaluate_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Insight Engine", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .source-tag {
        display: inline-block;
        margin: 4px;
        padding: 4px 12px;
        border-radius: 12px;
        background: #e1e7ff;
        color: #3b82f6;
        font-size: 0.8rem;
        border: 1px solid #3b82f6;
        font-weight: bold;
    }
    .view-link {
        font-size: 0.8rem;
        color: #10b981;
        text-decoration: none;
        font-weight: 600;
        margin-left: 5px;
    }
    .view-link:hover { 
        text-decoration: underline; 
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title(" System Panel")

    # Clear chat history button
    if st.button(" Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------- MAIN LAYOUT ----------------
st.title("Enterprise Chatbot")
# Two-column layout:
# - Left: Chat
# - Right: Evaluation & Sources
chat_col, eval_col = st.columns([1.2, 0.8], gap="large")


# ---------------- CHAT COLUMN ----------------
with chat_col:
    st.subheader(" Chat")
    chat_container = st.container(height=500)

    # Render previous messages
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


# ---------------- USER INPUT ----------------
question = st.chat_input("Ask a question...")

# ---------------- QUERY HANDLING ----------------
if question:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Run RAG pipeline
    with st.spinner("Processing..."):
        result = hybrid_rag_answer(question)
        metrics = evaluate_answer(
            question,
            result["answer"],
            result["retrieved_chunks"]
        )

    # Add assistant response with metadata
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "source_details": result.get("source_details", []),
        "has_sources": result["has_sources"],
        "metrics": metrics
    })

    # Rerun app to refresh UI
    st.rerun()

# ---------------- EVALUATION COLUMN ----------------
with eval_col:
    st.subheader(" Performance Analysis")

    # Only show evaluation if last message is assistant
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
        last_msg = st.session_state.chat_history[-1]

        # Tabs for sources and metrics
        tab1, tab2 = st.tabs([" Sources", " Evaluation Scores"])

        # ---------------- SOURCES TAB ----------------
        with tab1:
            st.markdown("#### Data Sources Used")

            if last_msg["has_sources"] and last_msg.get("source_details"):
                for s in last_msg["source_details"]:
                    file_url = f"app/static/{s['file']}"
                    print(file_url)

                    # Render source tag and link
                    st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <span class="source-tag">{s['name']}</span>
                            <a href="{file_url}" target="_blank" class="view-link">
                                Open File ↗
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No documents were relevant enough to be used.")

        # ---------------- METRICS TAB ----------------
        with tab2:
            st.markdown("#### RAG Quality Metrics")
            m = last_msg["metrics"]

            # ----- Retrieval Relevance -----
            rel_score = (m["retrieval_relevance"]["max_similarity"] if last_msg["has_sources"] else 0)
            st.write("**Retrieval Relevance** (How well chunks match question)")
            st.progress( max(0.0, min(1.0, rel_score)), text=f"Avg Relevance: {rel_score:.2f}")
            st.divider()

            # ----- Answer Grounding -----
            grd_score = (m["answer_grounding"]["avg_grounding"] if last_msg["has_sources"] else 0 )
            st.write("**Answer Grounding** (How well answer matches chunks)")
            st.progress(max(0.0, min(1.0, grd_score)),text=f"Avg Grounding: {grd_score:.2f}")
            st.divider()