#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("BAAI/bge-m3")
from pathlib import Path
cache = Path.home() / ".cache" / "huggingface"
for archivo in cache.rglob("modules.json"):
    if "bge-m3" in str(archivo).lower():
        print(archivo)