from sentence_transformers import SentenceTransformer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent

model = SentenceTransformer('all-mpnet-base-v2')
model.save(ROOT_DIR / 'data' / 'all-mpnet-base-v2')
