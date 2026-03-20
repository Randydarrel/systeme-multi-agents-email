import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from googletrans import Translator


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

translator = Translator()


client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("enron_emails")


query_fr = "réunion prévue la semaine prochaine concernant le trading de l'énergie"
query_embedding = embedding_model.encode(query_fr).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


print("\n🔍 Requête (FR) :", query_fr)
print("\n📄 Emails similaires (traduits en français) :\n")

for i, doc in enumerate(results["documents"][0]):
    traduction = translator.translate(doc, src="en", dest="fr").text

    print("=" * 80)
    print(f"Résultat {i+1}")
    print(traduction[:800])
