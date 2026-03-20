print("agent_redacteur module chargé avec succès.")

import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class agent_redacteur:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if api_key is None:
            raise ValueError("GROQ_API_KEY non trouvée. Vérifie le fichier .env.")

        self.client = Groq(api_key=api_key)

        print("Agent Rédacteur créé avec succès.")

    def generer_reponse(self, analyse, emails_similaires):
        """
        Génère une réponse professionnelle en français.
        """

        # Construction du contexte RAG
        contexte = ""
        for i, email in enumerate(emails_similaires[:2], 1):  # limité à 2 pour économiser tokens
            contexte += f"\nEmail {i}:\n{email['document']}\n"

        prompt = f"""
RÔLE :
Tu es un assistant expert en rédaction d’emails professionnels.



ANALYSE :
- Intention : {analyse['intention']}
- Ton : {analyse['ton']}
- Questions : {analyse['questions']}
- Entités clés : {analyse['entites']}

EMAILS SIMILAIRES (CONTEXTE) :
{contexte}

OBJECTIF :
Rédiger une réponse professionnelle complète.

RÈGLES IMPORTANTES :
- Répondre clairement à l’email
- Répondre à toutes les questions
- Ton professionnel et poli
- Maximum 120 mots
- Répondre uniquement en français
- Ne jamais utiliser de placeholders comme [Nom] ou [Destinataire]

STRUCTURE OBLIGATOIRE :

Bonjour,

[rédiger une réponse complète en plusieurs phrases]

Cordialement,
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )
        
        return response.choices[0].message.content.strip()