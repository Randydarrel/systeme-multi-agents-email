

import re
from typing import Dict, List


class agent_analyseur:
    

    def analyse(self, email_text: str) -> Dict:
        

        intention = self._detect_intention(email_text)
        tone = self._detect_tone(email_text)
        questions = self._extract_questions(email_text)
        entities = self._extract_entities(email_text)

        return {
            "intention": intention,
            "ton": tone,
            "questions": questions,
            "entites": entities,
        }

    

    def _detect_intention(self, text: str) -> str:
        text_lower = text.lower().strip()

    #  Cas spécial : "merci de + verbe" = DEMANDE
        if text_lower.startswith("merci de"):
            return "demande"

    #  Remerciement réel
        if any(expr in text_lower for expr in [
            "je vous remercie",
            "merci pour",
            "thank you for",
            "thanks for"
        ]):
            return "remerciement"

    #  Demande explicite
        if "?" in text_lower or any(word in text_lower for word in [
            "pouvez-vous",
            "pourriez-vous",
            "merci de",
            "please",
            "could you"
        ]):
            return "demande"

    #  Support
        if any(word in text_lower for word in [
            "problème", "souci", "incident", "erreur", "issue", "problem"
        ]):
            return "support"

    #  Planification
        if any(word in text_lower for word in [
            "organiser", "planifier", "fixer", "schedule", "arrange"
        ]):
            return "planification"

        return "information"

    def _detect_tone(self, text: str) -> str:
        text_lower = text.lower()

        if any(word in text_lower for word in [
            "urgent", "asap", "immédiatement"
        ]) or "!" in text:
            return "urgent"

        if any(word in text_lower for word in [
            "cordialement", "sincèrement", "bien à vous"
        ]):
            return "formel"

        if any(word in text_lower for word in [
            "merci", "thank you"
        ]):
            return "positif"

        return "neutre"

    def _extract_questions(self, text: str) -> List[str]:
        questions = re.findall(r"([^?.!]*\?)", text)
        return [q.strip() for q in questions]

    def _extract_entities(self, text: str) -> List[str]:
        keywords = ["meeting", "réunion", "energy", "énergie", "trading", "semaine","projet","contrat","facture","paiement","deadline","rapport","budget","mois"]
        found = []
        text_lower = text.lower()

        for kw in keywords:
            if kw in text_lower:
                found.append(kw)

        return list(set(found))
    
print("Agent Analyseur créé avec succès.")
