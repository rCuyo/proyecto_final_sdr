import chromadb
from sentence_transformers import SentenceTransformer

# Modelo local
model = SentenceTransformer(
    r"C:\Users\alfredo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)
# Base vectorial
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection("peliculas")

# Consulta
consulta = "películas sobre viajes en el tiempo"

embedding = model.encode(
    consulta,
    normalize_embeddings=True
).tolist()

resultado = collection.query(
    query_embeddings=[embedding],
    n_results=5
)

for meta in resultado["metadatas"][0]:
    print("Título:", meta["titulo"])
    print("Categoría:", meta["categoria"])
    print("-" * 40)