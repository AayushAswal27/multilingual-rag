"""Streamlit chat interface for the multilingual RAG system."""

import streamlit as st

from src.rag import RAGPipeline

st.set_page_config(
    page_title="Multilingual RAG",
    page_icon="📄",
    layout="centered",
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """
    Build the pipeline once and reuse it across every re-run.

    Streamlit re-executes this whole script on each interaction.
    Without caching, the embedding model and LLM client would be
    rebuilt on every message, making the app unusably slow.
    """
    return RAGPipeline()


def main() -> None:
    st.title("📄 Multilingual Document Assistant")
    st.caption(
        "Ask questions in Hindi, English, or other Indian languages. "
        "Answers come only from the indexed documents."
    )

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Could not start the assistant: {exc}")
        st.stop()

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