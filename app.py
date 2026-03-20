import streamlit as st
from orchestration.graph import build_graph
from datetime import datetime

# Métriques RAGAS — imports avec fallback silencieux
try:
    from sentence_transformers import SentenceTransformer, util
    import shutil, re as _re
    from pathlib import Path as _Path

    def _load_sbert():
        """Charge le modèle SBERT en gérant le cache corrompu (meta tensor)."""
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        try:
            m = SentenceTransformer(model_name, local_files_only=False, device="cpu")
            m.encode("test", normalize_embeddings=True)
            return m
        except (NotImplementedError, RuntimeError, OSError):
            # Cache corrompu → on nettoie et on retélécharge
            for base in [
                _Path.home() / ".cache" / "torch" / "sentence_transformers",
                _Path.home() / ".cache" / "huggingface" / "hub",
            ]:
                for item in base.glob(f"*{model_name.replace('/', '_')}*"):
                    shutil.rmtree(item, ignore_errors=True)
                for item in base.glob(f"*{model_name.replace('/', '--')}*"):
                    shutil.rmtree(item, ignore_errors=True)
            m = SentenceTransformer(model_name, local_files_only=False, device="cpu")
            m.encode("test", normalize_embeddings=True)
            return m

    _SBERT = _load_sbert()
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False
    _SBERT = None

def calculer_metriques_ragas(
    question: str,
    generation: str,
    emails_similaires: list
) -> dict:
    """
    Calcule les 3 métriques RAGAS sans dépendance externe lourde,
    en utilisant sentence-transformers (déjà installé pour le RAG).

    ── Fidélité (Faithfulness) ─────────────────────────────────────────────
    Mesure si chaque phrase de la réponse est soutenue par au moins
    un document récupéré. On encode chaque phrase de la réponse et
    chaque document, puis on calcule la similarité cosine max.
    Score = proportion de phrases de la réponse dont sim_max >= seuil.

    ── Pertinence de la réponse (Answer Relevancy) ──────────────────────────
    Mesure si la réponse répond bien à la question posée.
    On compare directement question ↔ réponse générée via similarité cosine.

    ── Exactitude de la réponse (Answer Correctness) ────────────────────────
    Compare la réponse générée avec les documents récupérés (ground truth
    approché). Moyenne des similarités cosine réponse ↔ chaque document.
    """
    if not METRICS_AVAILABLE or not generation.strip() or not emails_similaires:
        return {}
    try:
        model = _SBERT
        contextes = [em.get("document", "").strip() for em in emails_similaires if em.get("document","").strip()]
        if not contextes:
            return {}

        # ── Fidélité ────────────────────────────────────────────────────────
        # Découper la réponse en phrases (split sur . ! ?)
        import re
        phrases = [p.strip() for p in re.split(r"[.!?]", generation) if len(p.strip()) > 10]
        if not phrases:
            phrases = [generation]

        emb_phrases   = model.encode(phrases,   normalize_embeddings=True, show_progress_bar=False)
        emb_contextes = model.encode(contextes, normalize_embeddings=True, show_progress_bar=False)

        seuil = 0.35  # seuil de similarité pour considérer une phrase "supportée"
        phrases_supportees = 0
        for emb_p in emb_phrases:
            sims = util.cos_sim(emb_p, emb_contextes)[0]
            if float(sims.max()) >= seuil:
                phrases_supportees += 1
        fidelite = phrases_supportees / len(phrases)

        # ── Pertinence de la réponse ─────────────────────────────────────────
        emb_question = model.encode(question,   normalize_embeddings=True, show_progress_bar=False)
        emb_reponse  = model.encode(generation, normalize_embeddings=True, show_progress_bar=False)
        pertinence = float(util.cos_sim(emb_question, emb_reponse)[0][0])
        pertinence = max(0.0, min(1.0, pertinence))

        # ── Exactitude de la réponse ─────────────────────────────────────────
        sims_exactitude = util.cos_sim(emb_reponse, emb_contextes)[0]
        exactitude = float(sims_exactitude.mean())
        exactitude = max(0.0, min(1.0, exactitude))

        return {
            "Fidélité":              round(fidelite,   4),
            "Pertinence réponse":    round(pertinence, 4),
            "Exactitude réponse":    round(exactitude, 4),
        }
    except Exception:
        return {}



def calculer_metriques_ragas(
    question: str,
    generation: str,
    emails_similaires: list
) -> dict:
    """
    Calcule les 3 métriques RAGAS sans dépendance externe lourde,
    en utilisant sentence-transformers (déjà installé pour le RAG).

    ── Fidélité (Faithfulness) ─────────────────────────────────────────────
    Mesure si chaque phrase de la réponse est soutenue par au moins
    un document récupéré. On encode chaque phrase de la réponse et
    chaque document, puis on calcule la similarité cosine max.
    Score = proportion de phrases de la réponse dont sim_max >= seuil.

    ── Pertinence de la réponse (Answer Relevancy) ──────────────────────────
    Mesure si la réponse répond bien à la question posée.
    On compare directement question ↔ réponse générée via similarité cosine.

    ── Exactitude de la réponse (Answer Correctness) ────────────────────────
    Compare la réponse générée avec les documents récupérés (ground truth
    approché). Moyenne des similarités cosine réponse ↔ chaque document.
    """
    if not METRICS_AVAILABLE or not generation.strip() or not emails_similaires:
        return {}
    try:
        model = _SBERT
        contextes = [em.get("document", "").strip() for em in emails_similaires if em.get("document","").strip()]
        if not contextes:
            return {}

        # ── Fidélité ────────────────────────────────────────────────────────
        # Découper la réponse en phrases (split sur . ! ?)
        import re
        phrases = [p.strip() for p in re.split(r"[.!?]", generation) if len(p.strip()) > 10]
        if not phrases:
            phrases = [generation]

        emb_phrases   = model.encode(phrases,   normalize_embeddings=True, show_progress_bar=False)
        emb_contextes = model.encode(contextes, normalize_embeddings=True, show_progress_bar=False)

        seuil = 0.35  # seuil de similarité pour considérer une phrase "supportée"
        phrases_supportees = 0
        for emb_p in emb_phrases:
            sims = util.cos_sim(emb_p, emb_contextes)[0]
            if float(sims.max()) >= seuil:
                phrases_supportees += 1
        fidelite = phrases_supportees / len(phrases)

        # ── Pertinence de la réponse ─────────────────────────────────────────
        emb_question = model.encode(question,   normalize_embeddings=True, show_progress_bar=False)
        emb_reponse  = model.encode(generation, normalize_embeddings=True, show_progress_bar=False)
        pertinence = float(util.cos_sim(emb_question, emb_reponse)[0][0])
        pertinence = max(0.0, min(1.0, pertinence))

        # ── Exactitude de la réponse ─────────────────────────────────────────
        sims_exactitude = util.cos_sim(emb_reponse, emb_contextes)[0]
        exactitude = float(sims_exactitude.mean())
        exactitude = max(0.0, min(1.0, exactitude))

        return {
            "Fidélité":              round(fidelite,   4),
            "Pertinence réponse":    round(pertinence, 4),
            "Exactitude réponse":    round(exactitude, 4),
        }
    except Exception:
        return {}


def calculer_metriques_rag(generation: str, emails_similaires: list) -> dict:
    """
    Calcule les métriques entre la réponse générée et les emails
    similaires récupérés de la base ChromaDB (vraie évaluation RAG).

    Stratégie : on compare generation contre chaque email similaire,
    puis on garde la moyenne pour ROUGE-1, ROUGE-L et BERTScore F1
    (les plus pertinents pour ce système), et le max pour BLEU
    (on veut savoir si au moins un email de référence est bien couvert).
    """
    if not METRICS_AVAILABLE or not generation.strip() or not emails_similaires:
        return {}
    try:
        scorer  = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleu_scores, r1_scores, rl_scores = [], [], []
        references_bert, generations_bert = [], []

        for em in emails_similaires:
            ref = em.get("document", "").strip()
            if not ref:
                continue
            # BLEU
            bleu = sentence_bleu([ref.split()], generation.split(), smoothing_function=smoother)
            bleu_scores.append(bleu)
            # ROUGE
            rouge = scorer.score(ref, generation)
            r1_scores.append(rouge["rouge1"].fmeasure)
            rl_scores.append(rouge["rougeL"].fmeasure)
            # Pour BERTScore (batch)
            references_bert.append(ref)
            generations_bert.append(generation)

        if not bleu_scores:
            return {}

        # BERTScore sur tous les paires en une fois
        _, _, F1 = bert_score(generations_bert, references_bert, lang="fr", verbose=False)
        bert_scores = F1.tolist()

        return {
            "BLEU (max)":      round(max(bleu_scores), 4),
            "ROUGE-1 (moy)":   round(sum(r1_scores) / len(r1_scores), 4),
            "ROUGE-L (moy)":   round(sum(rl_scores) / len(rl_scores), 4),
            "BERTScore F1 (moy)": round(sum(bert_scores) / len(bert_scores), 4),
        }
    except Exception as e:
        return {}

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MultiAgent Email AI",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  CSS  (uniquement des styles statiques)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0e1117;
    --surface: #161b27;
    --card:    #1c2333;
    --border:  #2a3347;
    --accent:  #4f8ef7;
    --accent2: #7c5cfc;
    --green:   #22c55e;
    --amber:   #f59e0b;
    --red:     #ef4444;
    --text:    #e2e8f0;
    --muted:   #8892a4;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* Hero */
.hero { text-align:center; padding:2.5rem 1rem 1.5rem; margin-bottom:2rem; }
.hero h1 {
    font-family:'DM Serif Display',serif; font-size:2.8rem;
    background:linear-gradient(135deg,#4f8ef7,#7c5cfc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.4rem;
}
.hero p { color:#8892a4; font-size:1rem; margin:0; }

/* Section title */
.stitle {
    font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#4f8ef7; margin:0 0 0.8rem;
    font-family:'DM Sans',sans-serif;
}

/* Divider */
.divider { border:none; border-top:1px solid #2a3347; margin:2rem 0; }

/* Agent card box */
.agent-box {
    background:#1c2333; border:1px solid #2a3347; border-radius:14px;
    padding:1.4rem 1.6rem; margin-bottom:1rem; min-height:200px;
}
.agent-box-green { border-color:#22c55e44; }

/* Card header */
.agent-hdr {
    display:flex; align-items:center; gap:0.65rem;
    padding-bottom:0.85rem; margin-bottom:1rem; border-bottom:1px solid #2a3347;
}
.agent-hdr-icon {
    font-size:1.1rem; width:36px; height:36px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    border-radius:8px; background:linear-gradient(135deg,#4f8ef718,#7c5cfc18);
}
.agent-hdr-label { font-weight:600; font-size:0.95rem; color:#e2e8f0; display:block; }
.agent-hdr-sub   { font-size:0.74rem; color:#8892a4; display:block; }

/* Field label */
.flabel {
    font-size:0.63rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#8892a4; margin-bottom:0.4rem;
    margin-top:0.9rem; display:block;
}

/* Tags */
.tag-intent { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:2px; border:1px solid #4f8ef7; color:#4f8ef7; background:#4f8ef712; }
.tag-tone   { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:2px; border:1px solid #7c5cfc; color:#7c5cfc; background:#7c5cfc12; }
.tag-entity { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:2px; border:1px solid #f59e0b; color:#f59e0b; background:#f59e0b12; }

/* Question */
.q-row { display:flex; gap:8px; padding:6px 0; border-bottom:1px solid #2a3347; font-size:0.84rem; color:#e2e8f0; line-height:1.5; }
.q-row:last-child { border-bottom:none; }
.q-dot { color:#4f8ef7; font-weight:700; flex-shrink:0; }

/* RAG */
.rag-block { background:#161b27; border:1px solid #2a3347; border-radius:10px; padding:0.85rem 1rem; margin-bottom:0.7rem; }
.rag-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem; }
.rag-badge { font-size:0.7rem; font-weight:700; color:#4f8ef7; background:#4f8ef715; border:1px solid #4f8ef730; border-radius:6px; padding:2px 8px; }
.rag-pct   { font-size:0.72rem; color:#8892a4; }
.rag-pct strong { color:#e2e8f0; }
.rag-bar   { width:100%; height:4px; background:#2a3347; border-radius:99px; overflow:hidden; margin-bottom:0; }
.rag-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#4f8ef7,#7c5cfc); }

/* Response text */
.resp-body { font-size:0.9rem; line-height:1.85; color:#e2e8f0; white-space:pre-wrap; word-break:break-word; }

/* Final email */
.final-wrap { background:#07111f; border:1.5px solid #1e3a6e; border-radius:14px; padding:1.6rem 1.8rem; }
.final-top  { display:flex; align-items:center; gap:0.7rem; padding-bottom:0.9rem; margin-bottom:1rem; border-bottom:1px solid #1e3a6e; }
.final-ttl  { font-weight:700; font-size:1rem; color:#4f8ef7; display:block; }
.final-sub  { font-size:0.74rem; color:#8892a4; display:block; }
.final-body { font-size:0.93rem; line-height:1.9; color:#e2e8f0; white-space:pre-wrap; word-break:break-word; }

/* Metric */
.mrow  { display:flex; align-items:center; gap:0.85rem; margin-bottom:0.65rem; }
.mname { font-size:0.78rem; color:#8892a4; width:95px; flex-shrink:0; }
.mbar  { flex:1; height:6px; background:#2a3347; border-radius:99px; overflow:hidden; }
.mfill { height:100%; border-radius:99px; }
.mval  { font-size:0.82rem; font-weight:600; color:#e2e8f0; width:44px; text-align:right; flex-shrink:0; }

/* st.text() — ciblage exhaustif de tous les sélecteurs Streamlit possibles */
[data-testid="stText"],
[data-testid="stText"] p,
[data-testid="stText"] pre,
[data-testid="stText"] span,
.stText,
.stText p,
.stText pre,
.element-container div[data-testid="stText"] p,
.element-container div[data-testid="stText"] pre {
    background-color: #0b1120 !important;
    color: #ffffff !important;
    border: 1px solid #2a3347 !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    line-height: 1.65 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Textarea — actif ET désactivé */
.stTextArea textarea {
    background-color:#080d18 !important; border:1px solid #2a3347 !important;
    color:#e2e8f0 !important; border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.9rem !important;
    -webkit-text-fill-color:#e2e8f0 !important;
    opacity:1 !important;
}
.stTextArea textarea:disabled {
    color:#e2e8f0 !important;
    -webkit-text-fill-color:#e2e8f0 !important;
    opacity:1 !important;
    background-color:#0d1525 !important;
    cursor:default !important;
}
.stTextArea textarea:focus { border-color:#4f8ef7 !important; box-shadow:0 0 0 2px #4f8ef720 !important; }
.stTextArea label { color:#8892a4 !important; font-size:0.82rem !important; }

/* Buttons */
.stButton > button {
    border-radius:9px !important; font-family:'DM Sans',sans-serif !important;
    font-weight:600 !important; transition:all 0.2s !important;
}
.stButton > button:first-child {
    background:linear-gradient(135deg,#4f8ef7,#7c5cfc) !important;
    color:white !important; border:none !important;
}
.stButton > button:hover { opacity:0.85 !important; transform:translateY(-1px) !important; }

/* Expander */
.streamlit-expanderHeader { color:#e2e8f0 !important; font-size:0.82rem !important; }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STATE & GRAPH
# ─────────────────────────────────────────────
for key, default in {"historique": [], "result": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_graph():
    return build_graph()
graph_app = load_graph()

# ─────────────────────────────────────────────
#  HELPERS — tout le HTML dynamique est
#  construit en Python, puis passé en 1 seul appel
# ─────────────────────────────────────────────
def cosine_to_sim(dist: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(dist)))

def tone_icon(tone: str) -> str:
    return {"urgent": "🔴", "formel": "🔵", "positif": "🟢", "neutre": "⚪"}.get(tone, "⚪")

def render_agent_header(icon: str, label: str, sub: str) -> str:
    return (
        f'<div class="agent-hdr">'
        f'  <div class="agent-hdr-icon">{icon}</div>'
        f'  <div><span class="agent-hdr-label">{label}</span>'
        f'       <span class="agent-hdr-sub">{sub}</span></div>'
        f'</div>'
    )

def metric_bar_html(name: str, value: float, gradient: str) -> str:
    pct = min(max(value * 100, 0), 100)
    return (
        f'<div class="mrow">'
        f'  <div class="mname">{name}</div>'
        f'  <div class="mbar"><div class="mfill" style="width:{pct:.1f}%;background:{gradient};"></div></div>'
        f'  <div class="mval">{value:.3f}</div>'
        f'</div>'
    )

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✉ MultiAgent Email AI</h1>
    <p>Analyse · Récupération RAG · Rédaction · Vérification automatique</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT
# ─────────────────────────────────────────────
st.markdown('<p class="stitle">📨 Email entrant</p>', unsafe_allow_html=True)
email_input = st.text_area(
    label="Collez l'email à traiter :",
    height=160,
    placeholder="Ex : Bonjour, pouvez-vous confirmer votre disponibilité pour la réunion de la semaine prochaine ?"
)

col_gen, col_clr = st.columns(2)
with col_gen:
    generate_btn = st.button("⚡ Générer la réponse", use_container_width=True)
with col_clr:
    clear_btn = st.button("🗑 Effacer", use_container_width=True)

if clear_btn:
    st.session_state.result = None
    st.rerun()

if generate_btn:
    if not email_input.strip():
        st.warning("Veuillez saisir un email avant de générer.")
    else:
        with st.spinner("Les agents travaillent…"):
            result = graph_app.invoke({"email": email_input})

        # Calcul automatique des métriques RAGAS
        finale     = result.get("reponse_finale", "")
        emails_sim = result.get("emails_similaires", [])
        with st.spinner("Calcul des métriques RAGAS…"):
            metriques = calculer_metriques_ragas(email_input, finale, emails_sim)
        result["metriques"] = metriques

        st.session_state.result = result
        st.session_state.historique.append({
            "date":    datetime.now().strftime("%d/%m/%Y %H:%M"),
            "email":   email_input,
            "reponse": finale,
        })
        st.success("✅ Réponse générée avec succès !")

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
result = st.session_state.result

if result:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="stitle">🔍 Résultats des agents</p>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    #  ROW 1 : Analyseur  |  Récupérateur
    # ══════════════════════════════════════
    col_ana, col_rec = st.columns(2, gap="large")

    # ── Agent Analyseur ──────────────────
    with col_ana:
        analyse   = result.get("analyse", {})
        intention = analyse.get("intention", "—")
        ton       = analyse.get("ton", "—")
        questions = analyse.get("questions", [])
        entites   = analyse.get("entites", [])

        # Construire tout le HTML en Python avant st.markdown
        hdr = render_agent_header("🧠", "Agent Analyseur", "Intention · Ton · Entités · Questions")

        intent_html = f'<span class="tag-intent">{intention.upper()}</span>'

        tone_html = f'<span class="tag-tone">{tone_icon(ton)}&nbsp;{ton}</span>'

        if entites:
            entities_html = " ".join(f'<span class="tag-entity">{e}</span>' for e in entites)
        else:
            entities_html = '<span style="color:#8892a4;font-size:0.82rem;">Aucune</span>'

        if questions:
            q_rows = "".join(
                f'<div class="q-row"><span class="q-dot">›</span>'
                f'<span>{q.replace("<","&lt;").replace(">","&gt;")}</span></div>'
                for q in questions
            )
        else:
            q_rows = '<span style="color:#8892a4;font-size:0.82rem;">Aucune question détectée</span>'

        st.markdown(f"""
        <div class="agent-box">
            {hdr}
            <span class="flabel">Intention détectée</span>
            {intent_html}
            <span class="flabel">Ton de l'email</span>
            {tone_html}
            <span class="flabel">Entités clés ({len(entites)})</span>
            <div style="margin-top:4px;">{entities_html}</div>
            <span class="flabel">Questions détectées ({len(questions)})</span>
            <div style="margin-top:4px;">{q_rows}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Agent Récupérateur ───────────────
    with col_rec:
        emails_similaires = result.get("emails_similaires", [])
        hdr = render_agent_header("🔎", "Agent Récupérateur",
                                  f"{len(emails_similaires)} email(s) similaires — RAG")

        if not emails_similaires:
            st.markdown(
                f'<div class="agent-box">{hdr}'
                f'<span style="color:#8892a4;font-size:0.85rem;">Aucun email similaire récupéré.</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # En-tête de la carte agent (statique)
            st.markdown(f'<div class="agent-box">{hdr}</div>', unsafe_allow_html=True)

            for i, em in enumerate(emails_similaires, 1):
                dist       = em.get("distance", 1.0)
                similarity = cosine_to_sim(dist)
                pct        = similarity * 100
                doc        = em.get("document", "").strip()

                # Barre similarité — HTML 100% statique (pas de texte doc dedans)
                st.markdown(
                    f'<div class="rag-block">'
                    f'  <div class="rag-top">'
                    f'    <span class="rag-badge">Email #{i}</span>'
                    f'    <span class="rag-pct">Similarité : <strong>{similarity:.0%}</strong>'
                    f'    &nbsp;·&nbsp; distance = {dist:.3f}</span>'
                    f'  </div>'
                    f'  <div class="rag-bar"><div class="rag-fill" style="width:{pct:.1f}%;"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # Contenu cliquable — expander natif Streamlit, st.text = blanc garanti
                with st.expander(f"📄 Voir le contenu de l'email #{i}"):
                    st.text(doc)

    # ══════════════════════════════════════
    #  ROW 2 : Rédacteur  |  Vérificateur
    # ══════════════════════════════════════
    col_red, col_ver = st.columns(2, gap="large")

    reponse_brouillon = result.get("reponse_brouillon", "")
    reponse_finale    = result.get("reponse_finale", "")

    with col_red:
        hdr  = render_agent_header("✍️", "Agent Rédacteur", "Brouillon généré par le LLM")
        body = (
            f'<div class="resp-body">{reponse_brouillon}</div>'
            if reponse_brouillon
            else '<span style="color:#8892a4;font-style:italic;">Aucun brouillon disponible.</span>'
        )
        st.markdown(f'<div class="agent-box">{hdr}{body}</div>', unsafe_allow_html=True)

    with col_ver:
        hdr  = render_agent_header("✅", "Agent Vérificateur", "Version corrigée &amp; validée")
        body = (
            f'<div class="resp-body">{reponse_finale}</div>'
            if reponse_finale
            else '<span style="color:#8892a4;font-style:italic;">Aucune réponse vérifiée.</span>'
        )
        st.markdown(f'<div class="agent-box agent-box-green">{hdr}{body}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    #  EMAIL FINAL
    # ══════════════════════════════════════
    if reponse_finale:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="stitle">📤 Email final prêt à envoyer</p>', unsafe_allow_html=True)

        finale_esc = (reponse_finale
                      .replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))

        st.markdown(f"""
        <div class="final-wrap">
            <div class="final-top">
                <div style="font-size:1.5rem;">📧</div>
                <div>
                    <span class="final-ttl">Réponse finale validée</span>
                    <span class="final-sub">Générée et vérifiée par le système multi-agents</span>
                </div>
            </div>
            <div class="final-body">{finale_esc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════
    #  MÉTRIQUES RAGAS
    # ══════════════════════════════════════
    metriques = result.get("metriques", {})
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="stitle">📊 Évaluation RAGAS</p>', unsafe_allow_html=True)

    # Bandeau explicatif
    st.markdown(
        '<div style="background:#161b27;border:1px solid #2a3347;border-radius:10px;' +
        'padding:0.9rem 1.2rem;margin-bottom:1.2rem;font-size:0.83rem;color:#8892a4;line-height:1.75;">' +
        '<strong style="color:#e2e8f0;">📖 Framework RAGAS — 3 métriques clés</strong><br>' +
        '<strong style="color:#4f8ef7;">Fidélité</strong> — chaque affirmation de la réponse est-elle supportée ' +
        'par les documents récupérés ? (détecte les hallucinations)<br>' +
        '<strong style="color:#7c5cfc;">Pertinence réponse</strong> — la réponse répond-elle bien à la question posée ?<br>' +
        '<strong style="color:#f59e0b;">Exactitude réponse</strong> — la réponse est-elle cohérente avec ' +
        'les emails similaires de la base ChromaDB ?'  +
        '</div>',
        unsafe_allow_html=True
    )

    if not metriques:
        st.info("ℹ️ Métriques RAGAS non disponibles — vérifiez l'installation de sentence-transformers.")
    else:
        fidelite   = metriques.get("Fidélité", 0)
        pertinence = metriques.get("Pertinence réponse", 0)
        exactitude = metriques.get("Exactitude réponse", 0)

        def ragas_color(val: float) -> str:
            if val >= 0.7: return "#22c55e"
            if val >= 0.45: return "#f59e0b"
            return "#ef4444"

        def ragas_label(val: float) -> str:
            if val >= 0.7: return "Bon"
            if val >= 0.45: return "Modéré"
            return "Faible"

        def ragas_card(icon, title, subtitle, description, val, gradient):
            c = ragas_color(val)
            lbl = ragas_label(val)
            pct = min(max(val * 100, 0), 100)
            return (
                f'<div class="agent-box" style="text-align:center;">' +
                render_agent_header(icon, title, subtitle) +
                f'<div style="font-size:0.78rem;color:#8892a4;margin-bottom:1rem;line-height:1.5;">{description}</div>' +
                f'<div style="font-size:2.8rem;font-weight:700;color:{c};' +
                f'font-family:\'DM Serif Display\',serif;line-height:1;margin-bottom:0.3rem;">{val:.2f}</div>' +
                f'<div style="font-size:0.78rem;color:{c};font-weight:600;margin-bottom:1rem;">{lbl}</div>' +
                f'<div style="height:6px;background:#2a3347;border-radius:99px;overflow:hidden;">' +
                f'<div style="height:100%;width:{pct:.1f}%;background:{gradient};border-radius:99px;"></div>' +
                f'</div>' +
                f'</div>'
            )

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.markdown(ragas_card(
                "🔒", "Fidélité", "Faithfulness",
                "Proportion des affirmations de la réponse supportées par les documents récupérés. Score élevé = pas d'hallucination.",
                fidelite, "linear-gradient(90deg,#4f8ef7,#7c5cfc)"
            ), unsafe_allow_html=True)

        with c2:
            st.markdown(ragas_card(
                "🎯", "Pertinence", "Answer Relevancy",
                "Similarité sémantique entre la question initiale et la réponse générée. Score élevé = réponse bien ciblée.",
                pertinence, "linear-gradient(90deg,#7c5cfc,#a855f7)"
            ), unsafe_allow_html=True)

        with c3:
            st.markdown(ragas_card(
                "✅", "Exactitude", "Answer Correctness",
                "Cohérence entre la réponse générée et les emails similaires de la base (ground truth approché).",
                exactitude, "linear-gradient(90deg,#f59e0b,#f97316)"
            ), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HISTORIQUE — 100% natif, zéro texte dans HTML
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="stitle">🕐 Historique des réponses</p>', unsafe_allow_html=True)

if not st.session_state.historique:
    st.info("Aucune réponse générée pour le moment.")
else:
    for idx, item in enumerate(reversed(st.session_state.historique)):
        # En-tête carte — HTML 100% statique (seule la date vient de Python, pas du texte utilisateur)
        st.markdown(
            f'<div style="background:#1c2333;border:1px solid #2a3347;border-radius:13px;'
            f'padding:0.9rem 1.4rem 0.3rem;margin-bottom:0.4rem;">'
            f'<span style="font-size:0.72rem;color:#8892a4;">🕐 {item["date"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Label EMAIL REÇU
        st.markdown(
            '<span style="font-size:0.63rem;font-weight:600;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#8892a4;">📨 Email reçu</span>',
            unsafe_allow_html=True
        )
        # st.text() = rendu brut garanti, jamais interprété comme HTML
        st.text(item["email"])

        # Label RÉPONSE GÉNÉRÉE
        st.markdown(
            '<span style="font-size:0.63rem;font-weight:600;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#8892a4;">✉️ Réponse générée</span>',
            unsafe_allow_html=True
        )
        st.text(item["reponse"])

        st.markdown('<div style="margin-bottom:1rem;"></div>', unsafe_allow_html=True)