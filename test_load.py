import logging
logging.basicConfig(level=logging.WARNING)

from src.retriever import Retriever
from src.prompts import RAG_PROMPT, format_context

r = Retriever()
chunks = r.retrieve_relevant("Who is excluded from the scheme?")

messages = RAG_PROMPT.format_messages(
    language="English",
    context=format_context(chunks),
    question="Who is excluded from the scheme?",
)

for m in messages:
    print(f"\n{'=' * 70}\n{m.type.upper()}\n{'=' * 70}")
    print(m.content)