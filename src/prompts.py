"""Prompt templates for grounded multilingual answering."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a factual assistant answering questions about \
government scheme documents.

Rules you must follow:

1. Answer ONLY from the CONTEXT below. Do not use outside knowledge, \
even if you believe you know the answer.
2. If the CONTEXT does not contain the answer, say so plainly. Do not \
guess, infer beyond the text, or fill gaps.
3. Write your entire answer in {language}. This is the language the user \
asked in. Do not translate quoted figures or scheme names unnecessarily.
4. Be concise and specific. Cite figures, dates, and clause numbers \
exactly as they appear in the CONTEXT.
5. Do not mention "the context" or "the documents" in your answer. Just \
answer the question directly.

CONTEXT:
{context}"""

USER_PROMPT = "{question}"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]
)

NO_CONTEXT_MESSAGE = {
    "en": "I could not find information about this in the available documents.",
    "hi": "उपलब्ध दस्तावेज़ों में मुझे इसके बारे में जानकारी नहीं मिली।",
}


def format_context(chunks) -> str:
    """
    Render retrieved chunks into a numbered block for the prompt.

    Numbering lets the model reference specific passages, and keeps
    separate chunks from bleeding into each other as one wall of text.
    """
    return "\n\n".join(
        f"[{i + 1}] (Source: {c.citation})\n{c.content}"
        for i, c in enumerate(chunks)
    )