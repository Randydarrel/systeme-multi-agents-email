"""
Script d'évaluation du système multi-agents — Chapitre IV
==========================================================
Lance ce script depuis la racine du projet :
    python scripts/evaluer_systeme.py

Il génère :
    - resultats/resultats_bruts.json     → toutes les sorties agent par agent
    - resultats/metriques_resume.json    → tableaux prêts pour le mémoire
    - resultats/rapport_evaluation.txt   → rapport lisible directement
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Fix du chemin : ajoute la racine du projet au sys.path ──────────────────
# Remonte de 2 niveaux si le script est dans scripts/, sinon de 1 niveau
_script_path = Path(__file__).resolve()
# Cherche la racine en remontant jusqu'à trouver orchestration/
_root = _script_path.parent
for _ in range(4):
    if (_root / "orchestration").exists():
        break
    _root = _root.parent

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)  # aussi changer le cwd pour que les chemins relatifs fonctionnent

print(f"[INFO] Racine du projet : {_root}")
print(f"[INFO] sys.path[0]     : {sys.path[0]}")

# ── Imports métriques ────────────────────────────────────────────────────────
try:
    import torch
    from sentence_transformers import SentenceTransformer, util

    # Patch torch.load pour éviter l'erreur meta tensor (PyTorch 2.x)
    _orig_load = torch.load
    def _safe_load(f, *args, **kwargs):
        kwargs.pop("mmap", None)
        kwargs["weights_only"] = False
        kwargs["map_location"] = "cpu"
        return _orig_load(f, *args, **kwargs)
    torch.load = _safe_load

    SBERT = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        local_files_only=False,
    )
    SBERT.encode("test", normalize_embeddings=True)
    torch.load = _orig_load
    METRICS_OK = True
    print("[OK] Modèle SBERT chargé pour les métriques RAGAS.")
except Exception as e:
    METRICS_OK = False
    SBERT = None
    print(f"[AVERTISSEMENT] SBERT non disponible : {e}")
    print("  Les métriques RAGAS ne seront pas calculées.")

# ── Import du graphe ─────────────────────────────────────────────────────────
from orchestration.graph import build_graph

# ════════════════════════════════════════════════════════════════════════════
#  JEU DE 50 EMAILS DE TEST
#  Répartis en 5 intentions × 10 emails chacune
# ════════════════════════════════════════════════════════════════════════════
EMAILS_TEST = {

    # ── DEMANDE (10) ─────────────────────────────────────────────────────────
    "demande": [
        "Bonjour, pouvez-vous m'envoyer le rapport financier du trimestre en cours avant vendredi ?",
        "Pourriez-vous confirmer la liste des participants à la réunion de la semaine prochaine ?",
        "Merci de bien vouloir transmettre les documents contractuels au service juridique.",
        "Pouvez-vous me faire parvenir les dernières mises à jour concernant le projet énergie ?",
        "Pourriez-vous vérifier si le budget alloué au département trading a été validé ?",
        "Merci de m'indiquer les coordonnées du responsable en charge du dossier client.",
        "Pouvez-vous me confirmer la date limite de soumission du rapport annuel ?",
        "Pourriez-vous partager le compte rendu de la dernière réunion du comité de direction ?",
        "Merci de bien vouloir préparer une synthèse des résultats du mois de mars.",
        "Pouvez-vous vérifier l'état d'avancement du contrat avec le fournisseur principal ?",
    ],

    # ── PLANIFICATION (10) ───────────────────────────────────────────────────
    "planification": [
        "Je souhaiterais organiser une réunion la semaine prochaine pour discuter du projet. Êtes-vous disponible ?",
        "Pouvons-nous planifier un appel téléphonique pour jeudi matin afin de faire le point ?",
        "Je voudrais fixer un rendez-vous avec vous concernant la stratégie de déploiement du système.",
        "Serait-il possible de programmer une réunion d'équipe pour le 15 de ce mois ?",
        "Je voudrais organiser une session de travail pour finaliser le budget du prochain trimestre.",
        "Pourrions-nous planifier une revue de projet avec toutes les parties prenantes cette semaine ?",
        "Je souhaite mettre en place une réunion hebdomadaire pour suivre l'avancement des livrables.",
        "Pourriez-vous me proposer des créneaux disponibles pour une réunion de 30 minutes ?",
        "Je voudrais planifier une démonstration du prototype devant l'équipe de direction.",
        "Pouvons-nous arranger une visioconférence avec nos partenaires internationaux la semaine prochaine ?",
    ],

    # ── SUPPORT (10) ─────────────────────────────────────────────────────────
    "support": [
        "J'ai un problème avec l'accès à la plateforme de trading. Pouvez-vous m'aider ?",
        "Nous rencontrons un incident sur le système de facturation depuis ce matin.",
        "Il y a une erreur lors de la génération du rapport mensuel. Que faire ?",
        "Je n'arrive pas à me connecter à l'application. Mon mot de passe ne fonctionne plus.",
        "Nous avons un souci avec l'intégration des données dans le système ERP.",
        "Un problème est survenu lors du traitement des transactions de la journée.",
        "Je rencontre une erreur 404 sur le portail client depuis hier soir.",
        "Le système de messagerie interne est en panne depuis deux heures. C'est urgent.",
        "Nous avons un incident critique sur le serveur de production. Pouvez-vous intervenir ?",
        "Il y a une anomalie dans les calculs du tableau de bord financier.",
    ],

    # ── REMERCIEMENT (10) ────────────────────────────────────────────────────
    "remerciement": [
        "Je vous remercie pour votre aide précieuse lors de la préparation de la présentation.",
        "Merci pour votre réponse rapide concernant notre demande de partenariat.",
        "Je tenais à vous remercier pour la qualité du rapport que vous avez soumis.",
        "Merci pour votre soutien tout au long de ce projet complexe.",
        "Je vous remercie d'avoir pris le temps de répondre à toutes nos questions.",
        "Merci pour votre disponibilité et votre professionnalisme lors de notre réunion.",
        "Je tiens à vous exprimer ma gratitude pour l'excellent travail réalisé.",
        "Merci pour la transmission rapide des documents que nous attendions.",
        "Je vous remercie pour votre implication dans la résolution de l'incident.",
        "Merci d'avoir coordonné efficacement les différentes équipes sur ce dossier.",
    ],

    # ── INFORMATION (10) ─────────────────────────────────────────────────────
    "information": [
        "Je vous informe que la réunion du comité de direction est reportée au 20 de ce mois.",
        "Nous souhaitons vous informer que le contrat a été signé et prend effet immédiatement.",
        "Veuillez noter que les bureaux seront fermés le vendredi 14 juillet.",
        "Je vous transmets les nouvelles directives concernant la politique de télétravail.",
        "Nous vous informons que le système sera en maintenance ce week-end.",
        "Voici le compte rendu de la réunion du 5 mars pour votre information.",
        "Je vous confirme que votre demande de congés a été approuvée.",
        "Nous souhaitons vous informer d'un changement de procédure pour les remboursements.",
        "Veuillez trouver ci-joint le calendrier des prochaines échéances importantes.",
        "Je vous informe que notre nouveau responsable commercial prend ses fonctions lundi.",
    ],
}

# ════════════════════════════════════════════════════════════════════════════
#  FONCTIONS DE CALCUL MÉTRIQUES RAGAS
# ════════════════════════════════════════════════════════════════════════════

def calculer_ragas(question: str, generation: str, emails_similaires: list) -> dict:
    """
    Calcule les 3 métriques RAGAS :
    - Fidélité          : proportion de phrases de la réponse supportées par le contexte
    - Pertinence        : similarité cosine question ↔ réponse
    - Exactitude        : similarité cosine réponse ↔ moyenne des emails similaires
    """
    if not METRICS_OK or not generation.strip() or not emails_similaires:
        return {"fidelite": None, "pertinence": None, "exactitude": None}

    try:
        contextes = [
            em.get("document", "").strip()
            for em in emails_similaires
            if em.get("document", "").strip()
        ]
        if not contextes:
            return {"fidelite": None, "pertinence": None, "exactitude": None}

        # ── Fidélité ─────────────────────────────────────────────────────────
        phrases = [p.strip() for p in re.split(r"[.!?]", generation) if len(p.strip()) > 10]
        if not phrases:
            phrases = [generation]

        emb_phrases   = SBERT.encode(phrases,   normalize_embeddings=True, show_progress_bar=False)
        emb_contextes = SBERT.encode(contextes, normalize_embeddings=True, show_progress_bar=False)

        seuil = 0.35
        supportees = sum(
            1 for ep in emb_phrases
            if float(util.cos_sim(ep, emb_contextes).max()) >= seuil
        )
        fidelite = supportees / len(phrases)

        # ── Pertinence ───────────────────────────────────────────────────────
        emb_question = SBERT.encode(question,   normalize_embeddings=True, show_progress_bar=False)
        emb_reponse  = SBERT.encode(generation, normalize_embeddings=True, show_progress_bar=False)
        pertinence   = float(util.cos_sim(emb_question, emb_reponse)[0][0])
        pertinence   = max(0.0, min(1.0, pertinence))

        # ── Exactitude ───────────────────────────────────────────────────────
        sims       = util.cos_sim(emb_reponse, emb_contextes)[0]
        exactitude = float(sims.mean())
        exactitude = max(0.0, min(1.0, exactitude))

        return {
            "fidelite":   round(fidelite,   4),
            "pertinence": round(pertinence, 4),
            "exactitude": round(exactitude, 4),
        }

    except Exception as e:
        print(f"  [ERREUR métriques] {e}")
        return {"fidelite": None, "pertinence": None, "exactitude": None}


def evaluer_analyseur(intention_attendue: str, analyse: dict) -> dict:
    """Compare l'intention attendue à celle détectée par l'Agent Analyseur."""
    intention_detectee = analyse.get("intention", "")
    return {
        "intention_attendue":  intention_attendue,
        "intention_detectee":  intention_detectee,
        "intention_correcte":  intention_detectee == intention_attendue,
        "ton_detecte":         analyse.get("ton", ""),
        "nb_entites":          len(analyse.get("entites", [])),
        "nb_questions":        len(analyse.get("questions", [])),
    }


# ════════════════════════════════════════════════════════════════════════════
#  BOUCLE DE TEST PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

def lancer_tests():
    print("\n" + "="*60)
    print("ÉVALUATION DU SYSTÈME MULTI-AGENTS")
    print(f"Début : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*60)

    os.makedirs("resultats", exist_ok=True)
    graph_app = build_graph()

    resultats_bruts = []
    total = sum(len(v) for v in EMAILS_TEST.values())
    compteur = 0

    for intention_attendue, emails in EMAILS_TEST.items():
        print(f"\n── Intention : {intention_attendue.upper()} ({len(emails)} emails) ──")

        for email in emails:
            compteur += 1
            print(f"  [{compteur:02d}/{total}] {email[:60]}...")

            debut = time.time()
            try:
                result = graph_app.invoke({"email": email})
            except Exception as e:
                print(f"    [ERREUR] {e}")
                continue
            duree = round(time.time() - debut, 2)

            # Évaluation Agent Analyseur
            eval_analyseur = evaluer_analyseur(
                intention_attendue,
                result.get("analyse", {})
            )

            # Distances RAG
            emails_sim  = result.get("emails_similaires", [])
            distances   = [round(e.get("distance", 1.0), 4) for e in emails_sim]
            similarites = [round(max(0, 1 - d), 4) for d in distances]

            # Métriques RAGAS
            finale = result.get("reponse_finale", "")
            ragas  = calculer_ragas(email, finale, emails_sim)

            # Longueur de la réponse finale
            nb_mots = len(finale.split()) if finale else 0

            # Vérifications qualitatives simples
            a_bonjour     = "bonjour" in finale.lower()
            a_cordialement = any(w in finale.lower() for w in ["cordialement", "bien à vous", "sincèrement"])
            a_placeholder = bool(re.search(r"\[.+?\]", finale))

            entree = {
                "id":                 compteur,
                "intention_attendue": intention_attendue,
                "email":              email,
                "analyse":            result.get("analyse", {}),
                "eval_analyseur":     eval_analyseur,
                "distances_rag":      distances,
                "similarites_rag":    similarites,
                "reponse_brouillon":  result.get("reponse_brouillon", ""),
                "reponse_finale":     finale,
                "nb_mots_finale":     nb_mots,
                "a_bonjour":          a_bonjour,
                "a_cordialement":     a_cordialement,
                "a_placeholder":      a_placeholder,
                "ragas":              ragas,
                "duree_secondes":     duree,
            }
            resultats_bruts.append(entree)

            # Affichage rapide
            ic = "✓" if eval_analyseur["intention_correcte"] else "✗"
            sim_moy = round(sum(similarites)/len(similarites), 2) if similarites else 0
            print(f"    {ic} intention | sim_moy={sim_moy:.0%} | "
                  f"fidélité={ragas['fidelite']} | "
                  f"pertinence={ragas['pertinence']} | "
                  f"exactitude={ragas['exactitude']} | "
                  f"{nb_mots} mots | {duree}s")

    # ── Sauvegarde brute ─────────────────────────────────────────────────────
    with open("resultats/resultats_bruts.json", "w", encoding="utf-8") as f:
        json.dump(resultats_bruts, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] resultats/resultats_bruts.json sauvegardé ({len(resultats_bruts)} entrées)")

    return resultats_bruts


# ════════════════════════════════════════════════════════════════════════════
#  CALCUL DES STATISTIQUES — tableaux du Chapitre IV
# ════════════════════════════════════════════════════════════════════════════

def calculer_statistiques(resultats: list) -> dict:
    """Produit tous les chiffres nécessaires au Chapitre IV."""

    def moy(lst):
        lst = [x for x in lst if x is not None]
        return round(sum(lst) / len(lst), 4) if lst else None

    def mini(lst):
        lst = [x for x in lst if x is not None]
        return round(min(lst), 4) if lst else None

    def maxi(lst):
        lst = [x for x in lst if x is not None]
        return round(max(lst), 4) if lst else None

    # ── Agent Analyseur ───────────────────────────────────────────────────────
    nb_total         = len(resultats)
    nb_correct       = sum(1 for r in resultats if r["eval_analyseur"]["intention_correcte"])
    precision_intent = round(nb_correct / nb_total, 4) if nb_total else 0

    # Précision par intention
    par_intention = {}
    for intention in EMAILS_TEST:
        sous = [r for r in resultats if r["intention_attendue"] == intention]
        bons = sum(1 for r in sous if r["eval_analyseur"]["intention_correcte"])
        par_intention[intention] = {
            "total":     len(sous),
            "corrects":  bons,
            "precision": round(bons / len(sous), 4) if sous else 0,
        }

    # Tons détectés
    tons = [r["eval_analyseur"]["ton_detecte"] for r in resultats]
    dist_tons = {t: tons.count(t) for t in set(tons)}

    # ── RAG ───────────────────────────────────────────────────────────────────
    toutes_distances   = [d for r in resultats for d in r["distances_rag"]]
    toutes_similarites = [s for r in resultats for s in r["similarites_rag"]]

    # ── Métriques RAGAS ───────────────────────────────────────────────────────
    fidelites   = [r["ragas"]["fidelite"]   for r in resultats]
    pertinences = [r["ragas"]["pertinence"] for r in resultats]
    exactitudes = [r["ragas"]["exactitude"] for r in resultats]

    # RAGAS par intention
    ragas_par_intention = {}
    for intention in EMAILS_TEST:
        sous = [r for r in resultats if r["intention_attendue"] == intention]
        ragas_par_intention[intention] = {
            "fidelite_moy":   moy([r["ragas"]["fidelite"]   for r in sous]),
            "pertinence_moy": moy([r["ragas"]["pertinence"] for r in sous]),
            "exactitude_moy": moy([r["ragas"]["exactitude"] for r in sous]),
        }

    # ── Qualité des réponses ──────────────────────────────────────────────────
    nb_mots      = [r["nb_mots_finale"] for r in resultats if r["nb_mots_finale"] > 0]
    nb_bonjour   = sum(1 for r in resultats if r["a_bonjour"])
    nb_cordial   = sum(1 for r in resultats if r["a_cordialement"])
    nb_placeholder = sum(1 for r in resultats if r["a_placeholder"])

    # ── Performances ─────────────────────────────────────────────────────────
    durees = [r["duree_secondes"] for r in resultats]

    stats = {
        "metadata": {
            "date":          datetime.now().strftime("%d/%m/%Y %H:%M"),
            "nb_emails_test": nb_total,
        },
        "agent_analyseur": {
            "precision_globale":  precision_intent,
            "nb_corrects":        nb_correct,
            "nb_total":           nb_total,
            "precision_par_intention": par_intention,
            "distribution_tons":  dist_tons,
        },
        "rag": {
            "distance_moy":   moy(toutes_distances),
            "distance_min":   mini(toutes_distances),
            "distance_max":   maxi(toutes_distances),
            "similarite_moy": moy(toutes_similarites),
            "similarite_min": mini(toutes_similarites),
            "similarite_max": maxi(toutes_similarites),
        },
        "ragas": {
            "fidelite":   {"moy": moy(fidelites),   "min": mini(fidelites),   "max": maxi(fidelites)},
            "pertinence": {"moy": moy(pertinences), "min": mini(pertinences), "max": maxi(pertinences)},
            "exactitude": {"moy": moy(exactitudes), "min": mini(exactitudes), "max": maxi(exactitudes)},
            "par_intention": ragas_par_intention,
        },
        "qualite_reponses": {
            "nb_mots_moy":        moy(nb_mots),
            "nb_mots_min":        mini(nb_mots),
            "nb_mots_max":        maxi(nb_mots),
            "pct_avec_bonjour":   round(nb_bonjour / nb_total, 4),
            "pct_avec_cordial":   round(nb_cordial / nb_total, 4),
            "pct_avec_placeholder": round(nb_placeholder / nb_total, 4),
        },
        "performance": {
            "duree_moy_s": moy(durees),
            "duree_min_s": mini(durees),
            "duree_max_s": maxi(durees),
        },
    }

    return stats


# ════════════════════════════════════════════════════════════════════════════
#  RAPPORT TEXTE — copier-coller dans le mémoire
# ════════════════════════════════════════════════════════════════════════════

def generer_rapport(stats: dict) -> str:
    s  = stats
    an = s["agent_analyseur"]
    rg = s["rag"]
    ra = s["ragas"]
    qr = s["qualite_reponses"]
    pf = s["performance"]

    lignes = [
        "=" * 65,
        "RAPPORT D'ÉVALUATION — SYSTÈME MULTI-AGENTS",
        f"Date : {s['metadata']['date']}  |  Emails testés : {s['metadata']['nb_emails_test']}",
        "=" * 65,

        "",
        "── SECTION IV.2 : AGENT ANALYSEUR ─────────────────────────────",
        f"Précision globale (intention) : {an['precision_globale']*100:.1f}%",
        f"  ({an['nb_corrects']}/{an['nb_total']} emails correctement classifiés)",
        "",
        "Précision par intention :",
    ]

    for intent, v in an["precision_par_intention"].items():
        lignes.append(f"  {intent:15s} : {v['precision']*100:.1f}%  ({v['corrects']}/{v['total']})")

    lignes += [
        "",
        "Distribution des tons détectés :",
    ]
    for ton, nb in sorted(an["distribution_tons"].items(), key=lambda x: -x[1]):
        lignes.append(f"  {ton:10s} : {nb} emails")

    lignes += [
        "",
        "── SECTION IV.3 : PIPELINE RAG ────────────────────────────────",
        f"Distance cosine moyenne (fr→en) : {rg['distance_moy']} "
        f"[{rg['distance_min']} – {rg['distance_max']}]",
        f"Similarité moyenne              : {rg['similarite_moy']*100:.1f}% "
        f"[{rg['similarite_min']*100:.1f}% – {rg['similarite_max']*100:.1f}%]",

        "",
        "── SECTION IV.4 : MÉTRIQUES RAGAS ─────────────────────────────",
        f"{'Métrique':<22} {'Moy':>6}  {'Min':>6}  {'Max':>6}",
        f"{'-'*44}",
        f"{'Fidélité':<22} {ra['fidelite']['moy']:>6.3f}  {ra['fidelite']['min']:>6.3f}  {ra['fidelite']['max']:>6.3f}",
        f"{'Pertinence réponse':<22} {ra['pertinence']['moy']:>6.3f}  {ra['pertinence']['min']:>6.3f}  {ra['pertinence']['max']:>6.3f}",
        f"{'Exactitude réponse':<22} {ra['exactitude']['moy']:>6.3f}  {ra['exactitude']['min']:>6.3f}  {ra['exactitude']['max']:>6.3f}",

        "",
        "RAGAS par intention :",
        f"{'Intention':<15} {'Fidél.':>7} {'Pertin.':>8} {'Exact.':>8}",
        f"{'-'*42}",
    ]

    for intent, v in ra["par_intention"].items():
        lignes.append(
            f"{intent:<15} {str(v['fidelite_moy']):>7}  {str(v['pertinence_moy']):>7}  {str(v['exactitude_moy']):>7}"
        )

    lignes += [
        "",
        "── SECTION IV.5 : QUALITÉ DES RÉPONSES ────────────────────────",
        f"Longueur moyenne      : {qr['nb_mots_moy']} mots [{qr['nb_mots_min']}–{qr['nb_mots_max']}]",
        f"Structure 'Bonjour'   : {qr['pct_avec_bonjour']*100:.1f}%",
        f"Structure 'Cordial.'  : {qr['pct_avec_cordial']*100:.1f}%",
        f"Avec placeholders     : {qr['pct_avec_placeholder']*100:.1f}%  (objectif = 0%)",

        "",
        "── PERFORMANCE ─────────────────────────────────────────────────",
        f"Durée moyenne par email : {pf['duree_moy_s']}s "
        f"[{pf['duree_min_s']}s – {pf['duree_max_s']}s]",

        "",
        "=" * 65,
    ]

    return "\n".join(lignes)


# ════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Lancer les tests
    resultats = lancer_tests()

    # 2. Calculer les statistiques
    print("\n[Calcul des statistiques...]")
    stats = calculer_statistiques(resultats)

    # 3. Sauvegarder le JSON de métriques
    with open("resultats/metriques_resume.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[OK] resultats/metriques_resume.json sauvegardé")

    # 4. Générer et sauvegarder le rapport texte
    rapport = generer_rapport(stats)
    with open("resultats/rapport_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(rapport)
    print("[OK] resultats/rapport_evaluation.txt sauvegardé")

    # 5. Afficher le rapport dans le terminal
    print("\n" + rapport)