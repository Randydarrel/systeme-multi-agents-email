import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

os.makedirs(CHROMA_PATH, exist_ok=True)

df = pd.read_csv(os.path.join(BASE_DIR, "data", "email_clean.csv"))

texts    = df["clean_message"].tolist()
metadatas = [{"file": f} for f in df["file"].tolist()]
ids      = [str(i) for i in range(len(texts))]

print("Nombre d'emails à indexer :", len(texts))

# ── Nouveau modèle multilingue ──────────────────────────────────────────────
# Remplace all-MiniLM-L6-v2 (anglais uniquement) par le modèle multilingue
# qui gère le français, l'anglais et 50+ langues dans le même espace vectoriel.
# Dimension de sortie : 384 (identique à l'ancien modèle → pas de changement ChromaDB).
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# ── IMPORTANT : on supprime l'ancienne collection avant de recréer ──────────
# Les embeddings de all-MiniLM-L6-v2 sont incompatibles avec le nouveau modèle
# même si la dimension (384) est identique — l'espace vectoriel est différent.
try:
    client.delete_collection(name="enron_emails")
    print("Ancienne collection supprimée.")
except Exception:
    print("Aucune collection existante à supprimer.")

collection = client.get_or_create_collection(name="enron_emails")

# ── Indexation par batch pour accélérer ────────────────────────────────────
BATCH_SIZE = 256

for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Indexation"):
    end        = min(start + BATCH_SIZE, len(texts))
    batch_text = texts[start:end]
    batch_meta = metadatas[start:end]
    batch_ids  = ids[start:end]

    # normalize_embeddings=True → distances cosine dans [0, 1] (0 = identique)
    embeddings = embedding_model.encode(
        batch_text,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.add(
        documents=batch_text,
        metadatas=batch_meta,
        ids=batch_ids,
        embeddings=embeddings
    )

print(f" Base ChromaDB créée avec {len(texts)} emails .")