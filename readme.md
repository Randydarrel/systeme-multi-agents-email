# Système Multi-Agents pour la Génération de Réponses aux Emails

Projet tutoré — Master 2 Data Science  
**Étudiant :** KWETCHE FOKAM Darrel Randy  
  
**Année :** 2025–2026

## Description

Système multi-agents basé sur LLM pour générer automatiquement des réponses
professionnelles aux emails, en s'appuyant sur une approche RAG avec le
Enron Email Dataset.

## Architecture

- **Agent Analyseur** — Détection intention, ton, entités
- **Agent Récupérateur** — RAG avec ChromaDB
- **Agent Rédacteur** — Génération LLaMA-3.3-70B via Groq
- **Agent Vérificateur** — Contrôle qualité LLM
- **Orchestration** — LangGraph
- **Interface** — Streamlit

## Installation
```bash
# 1. Cloner le repo
git clone https://github.com/Randydarrel/systeme-multi-agents-email.git
cd systeme-multi-agents-email

# 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows
# ou : source venv/bin/activate  (Linux/Mac)

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env
# Editer .env et ajouter ta GROQ_API_KEY

# 5. Indexer la base vectorielle
python scripts/create_embeddings.py

# 6. Lancer l'application
streamlit run app.py
```

## Variables d'environnement

Créer un fichier `.env` à la racine :
```
GROQ_API_KEY=ta_clé_groq_ici
```

## Technologies

| Composant | Technologie |
|-----------|-------------|
| LLM | LLaMA-3.3-70B-Versatile (Groq) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Base vectorielle | ChromaDB |
| Orchestration | LangGraph |
| Interface | Streamlit |
| Évaluation | RAGAS (custom) |