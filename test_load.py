import logging
logging.basicConfig(level=logging.INFO)

from src.embeddings import get_embeddings

emb = get_embeddings()

vec = emb.embed_query("Who is eligible for the PM-KISAN scheme?")
print(f"Dimensions: {len(vec)}")
print(f"First 5 values: {vec[:5]}")

# Cross-lingual sanity check
import numpy as np

en = np.array(emb.embed_query("Which farmers can receive money under this scheme?"))
hi = np.array(emb.embed_query("इस योजना के तहत किन किसानों को पैसा मिल सकता है?"))
unrelated = np.array(emb.embed_query("How do I bake a chocolate cake?"))

print(f"\nEN vs HI (same meaning):  {en @ hi:.4f}")
print(f"EN vs unrelated:         {en @ unrelated:.4f}")