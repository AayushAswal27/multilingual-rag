import logging
logging.basicConfig(level=logging.WARNING)

from src.retriever import Retriever
from src.prompts import RAG_PROMPT, format_context
from src.llm import get_llm

r = Retriever()
llm = get_llm()

question = "Who is excluded from the scheme?"
chunks = r.retrieve_relevant(question)

messages = RAG_PROMPT.format_messages(
    language="English",
    context=format_context(chunks),
    question=question,
)

response = llm.invoke(messages)

print(f"QUESTION: {question}\n")
print(f"ANSWER:\n{response.content}\n")
print("SOURCES:")
for c in chunks:
    print(f"  - {c.citation}  (distance {c.distance:.3f})")