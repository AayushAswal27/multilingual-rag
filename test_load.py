import logging
logging.basicConfig(level=logging.WARNING)

from src.retriever import Retriever

r = Retriever()

for query in [
    "Who is excluded from the scheme?",
    "इस योजना से कौन बाहर रखा गया है?",
    "How do I bake a chocolate cake?",
]:
    print(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")
    for c in r.retrieve(query):
        print(f"\n[{c.distance:.4f}] {c.citation}")
        print(c.content[:200].replace("\n", " "))

print("\n\nWith threshold (max_distance=1.6):")
print(f"Farming query kept: {len(r.retrieve_relevant('Who is excluded from the scheme?'))} chunks")
print(f"Hindi query kept:   {len(r.retrieve_relevant('इस योजना से कौन बाहर रखा गया है?'))} chunks")
print(f"Cake query kept:    {len(r.retrieve_relevant('How do I bake a chocolate cake?'))} chunks")