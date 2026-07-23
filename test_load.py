import logging
logging.basicConfig(level=logging.INFO)

from src.loaders import load_documents
from src.chunking import chunk_documents

docs = load_documents()
chunks = chunk_documents(docs)

print(f"\n{len(docs)} pages -> {len(chunks)} chunks")
print(f"\nAvg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)}")
print(f"\n--- Chunk 5 ---\n{chunks[5].page_content}")
print(f"\nMetadata: {chunks[5].metadata}")