"""Streamlit chat interface for the multilingual RAG system."""

from pathlib import Path

import streamlit as st

from src.loaders import load_uploaded_file
from src.chunking import chunk_documents
from src.vector_store import build_vector_store
from src.retriever import Retriever
from src.rag import RAGPipeline

st.set_page_config(
    page_title="Multilingual RAG",
    page_icon="📄",
    layout="centered",
)


@st.cache_resource
def get_default_pipeline() -> RAGPipeline:
    """
    Pipeline backed by the pre-built index on disk (PM-KISAN).

    Cached so the embedding model and LLM client are built once and
    reused across Streamlit's re-runs.
    """
    return RAGPipeline()


def build_pipeline_from_upload(file) -> RAGPipeline:
    """Build a pipeline from a freshly uploaded file, in memory."""
    suffix = Path(file.name).suffix
    docs = load_uploaded_file(file, suffix)
    chunks = chunk_documents(docs)
    store = build_vector_store(chunks)
    retriever = Retriever(store=store)
    return RAGPipeline(retriever=retriever)


def main() -> None:
    st.title("📄 Multilingual Document Assistant")
    st.caption(
        "Ask questions in Hindi, English, or other Indian languages. "
        "Answers come only from the indexed documents."
    )

    uploaded = st.file_uploader(
        "Upload a document to ask about (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
    )

    if uploaded is not None:
        # Only re-index when a genuinely new file appears — otherwise
        # every re-run would re-embed the whole document.
        if st.session_state.get("uploaded_name") != uploaded.name:
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    st.session_state.pipeline = build_pipeline_from_upload(uploaded)
                    st.session_state.uploaded_name = uploaded.name
                    st.session_state.messages = []  # fresh chat for a new doc
                except Exception as exc:
                    st.error(f"Could not index that file: {exc}")
                    st.stop()
        pipeline = st.session_state.pipeline
        st.caption(f"Answering from: **{uploaded.name}**")
    else:
        try:
            pipeline = get_default_pipeline()
        except Exception as exc:
            st.error(f"Could not start the assistant: {exc}")
            st.stop()
        st.caption("Answering from: **PM-KISAN guidelines** (default)")

    # Chat history lives in session state — survives re-runs
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history on every re-run
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    # Input box pinned to the bottom
    prompt = st.chat_input("Ask a question about the documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = pipeline.answer(prompt)

            st.markdown(response.answer)

            source_labels = [s.citation for s in response.sources]

            if response.grounded:
                st.caption(f"Language: {response.language}")
                if source_labels:
                    with st.expander("Sources"):
                        for label in source_labels:
                            st.markdown(f"- {label}")
            else:
                st.caption("No matching content found in the documents.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": source_labels,
        })


if __name__ == "__main__":
    main()