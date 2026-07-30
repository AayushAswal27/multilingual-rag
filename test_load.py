import logging
logging.basicConfig(level=logging.WARNING)

from src.rag import RAGPipeline

pipeline = RAGPipeline()

questions = [
    "Who is excluded from the scheme?",
    "इस योजना के तहत कितने पैसे मिलते हैं?",
    "योजना से कौन बाहर रखा गया है?",
    "How do I bake a chocolate cake?",
]

for q in questions:
    r = pipeline.answer(q)
    print(f"\n{'=' * 70}")
    print(f"Q [{r.language}]: {q}")
    print(f"{'=' * 70}")
    print(f"{r.answer}\n")
    if r.grounded:
        for s in r.sources:
            print(f"  · {s.citation}")
    else:
        print("  (no relevant context found)")