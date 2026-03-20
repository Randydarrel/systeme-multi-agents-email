import os
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")


client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

print("Collections disponibles :", client.list_collections())

collection = client.get_collection("enron_emails")
print("Nombre de documents :", collection.count())
