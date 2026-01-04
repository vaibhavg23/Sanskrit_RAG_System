"""
Configuration settings for Sanskrit RAG System
"""

import os

# Model Settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "google/flan-t5-base"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Retrieval Settings
CHUNK_SIZE = 500
TOP_K_DOCS = 3
EMBEDDING_DIMENSION = 384

# Generation Settings
MAX_LENGTH = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

# Performance
BATCH_SIZE = 16
USE_CACHE = True