"""Central configuration. Every path and constant lives here."""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths — resolved relative to project root, not the current working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
VECTORDB_DIR = ROOT_DIR / "vectordb"

# Embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 6

# LLM
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}