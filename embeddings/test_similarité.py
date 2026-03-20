import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("enron_emails")


query = "meeting scheduled for next week about energy trading"

query_embedding = embedding_model.encode(query).tolist()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


print("\n Requête :", query)
print("\n Emails les plus similaires :\n")

for i, doc in enumerate(results["documents"][0]):
    print("=" * 80)
    print(f"Résultat {i+1}")
    print(doc[:600])  # aperçu du contenu
