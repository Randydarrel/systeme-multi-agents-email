from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
import shutil

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _charger_modele() -> SentenceTransformer:
    """
    Charge le modèle sentence-transformers en gérant le cas du cache corrompu.

    L'erreur NotImplementedError / meta tensor survient quand le dossier
    de cache HuggingFace existe mais contient des fichiers incomplets ou
    corrompus (téléchargement interrompu, version PyTorch incompatible...).

    Solution : on tente de charger, et si ça échoue on supprime le cache
    et on retélécharge proprement.
    """
    import torch
    from pathlib import Path

    # Dossier de cache HuggingFace par défaut (Windows + Linux/Mac)
    hf_cache = Path(os.environ.get(
        "SENTENCE_TRANSFORMERS_HOME",
        os.path.join(Path.home(), ".cache", "torch", "sentence_transformers")
    ))
    model_cache = hf_cache / MODEL_NAME.replace("/", "_")

    def _load(ignore_cache: bool) -> SentenceTransformer:
        return SentenceTransformer(
            MODEL_NAME,
            local_files_only=False,
            device="cpu",          # force CPU pour éviter les conflits CUDA/meta
        )

    # Premier essai : chargement normal
    try:
        model = _load(ignore_cache=False)
        # Vérification rapide : on encode une phrase de test
        model.encode("test", normalize_embeddings=True)
        print(f"[AgentRecuperateur] Modèle '{MODEL_NAME}' chargé avec succès.")
        return model

    except (NotImplementedError, RuntimeError, OSError) as e:
        print(f"[AgentRecuperateur] Cache corrompu détecté ({type(e).__name__}). Nettoyage en cours…")

        # Supprimer le cache corrompu
        for cache_dir in [
            model_cache,
            hf_cache / f"sentence-transformers_{MODEL_NAME}",
        ]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"[AgentRecuperateur] Cache supprimé : {cache_dir}")

        # Vider aussi le cache HuggingFace Hub si présent
        hf_hub_cache = Path(os.environ.get(
            "HF_HOME",
            os.path.join(Path.home(), ".cache", "huggingface")
        )) / "hub"
        for item in hf_hub_cache.glob(f"*{MODEL_NAME.replace('/', '--')}*"):
            shutil.rmtree(item, ignore_errors=True)
            print(f"[AgentRecuperateur] Cache HF Hub supprimé : {item}")

        # Deuxième essai après nettoyage
        print("[AgentRecuperateur] Re-téléchargement du modèle…")
        model = _load(ignore_cache=False)
        model.encode("test", normalize_embeddings=True)
        print(f"[AgentRecuperateur] Modèle re-téléchargé avec succès.")
        return model


class AgentRecuperateur:

    def __init__(self, persist_dir: str = "embeddings/chroma_db"):
        self.embedding_model = _charger_modele()

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="enron_emails"
        )

        print("Agent Récupérateur (multilingue) prêt.")

    def recupere(self, analysis: Dict, top_k: int = 3) -> List[Dict]:
        """
        Récupère les top_k emails les plus similaires à l'analyse.
        """
        query_text = self._build_query(analysis)

        query_embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_emails = []
        for i in range(len(results["documents"][0])):
            retrieved_emails.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved_emails

    def _build_query(self, analysis: Dict) -> str:
        """
        Construit la requête textuelle depuis l'analyse de l'email.
        """
        parts = []

        intention = analysis.get("intention", "")
        ton       = analysis.get("ton", "")
        entites   = analysis.get("entites", [])
        questions = analysis.get("questions", [])

        if intention:
            parts.append(intention)
        if ton:
            parts.append(ton)
        if entites:
            parts.extend(entites)
        if questions:
            parts.extend(questions)

        query = " ".join(parts).strip()
        return query if query else "email professionnel"


print("Agent Récupérateur créé avec succès.")