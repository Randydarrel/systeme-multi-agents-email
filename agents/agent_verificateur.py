print("agent_verificateur module chargé avec succès.")

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class agent_verificateur:
    """
    Agent chargé de vérifier ET réécrire la réponse générée.
    Il produit TOUJOURS une version distincte du brouillon,
    même quand celui-ci est de bonne qualité.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY non trouvée dans le .env")
        self.client = Groq(api_key=api_key)
        print("Agent Vérificateur créé avec succès.")

    def verifier_et_ameliorer(self, analyse: dict, reponse: str) -> str:

        prompt = f"""
RÔLE :
Tu es un contrôleur qualité expert en communication professionnelle.
Tu dois OBLIGATOIREMENT produire une version réécrite de la réponse,
même si elle te semble correcte. Ton rôle est d'améliorer, enrichir
et affiner — pas de recopier.

ANALYSE DE L'EMAIL ORIGINAL :
- Intention : {analyse['intention']}
- Ton attendu : {analyse['ton']}
- Questions à traiter : {analyse['questions']}
- Entités clés : {analyse['entites']}

BROUILLON À AMÉLIORER :
{reponse}

TÂCHES OBLIGATOIRES :
1. Vérifier que toutes les questions sont bien traitées
2. Enrichir le contenu avec une phrase de valeur ajoutée
3. Affiner le style pour le rendre plus naturel et professionnel
4. Varier le vocabulaire par rapport au brouillon
5. S'assurer que la réponse est complète et engageante

RÈGLES ABSOLUES :
- TOUJOURS produire une version différente du brouillon (jamais une copie)
- Ne jamais utiliser de placeholders comme [Nom] ou [Destinataire]
- Répondre uniquement en français
- Maximum 120 mots
- Ne jamais inventer d'informations factuelles

FORMAT OBLIGATOIRE :

Bonjour,

[version améliorée et distincte du brouillon — obligatoirement différente]

Cordialement,
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.6,  # légèrement plus élevé pour forcer la variation
        )

        resultat = response.choices[0].message.content.strip()

        # Garde-fou : si le vérificateur renvoie exactement le même texte,
        # on le force à reformuler avec une instruction plus directe
        if resultat.strip() == reponse.strip():
            prompt_force = f"""
La réponse suivante doit être réécrite différemment, en conservant le même sens
mais avec un vocabulaire et une structure différents :

{reponse}

Règles : français uniquement, max 120 mots, format Bonjour / [contenu] / Cordialement.
Produis UNIQUEMENT la version réécrite, sans aucune explication.
"""
            response2 = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_force}],
                max_tokens=200,
                temperature=0.7,
            )
            resultat = response2.choices[0].message.content.strip()

        return resultat