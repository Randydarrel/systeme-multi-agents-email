from agent_analyseur import agent_analyseur
from agent_recuperateur import AgentRecuperateur
from agent_redacteur import agent_redacteur


def test_agent_redacteur_llm():
    email = """
    Une réunion est prévue la semaine prochaine concernant le trading de l'énergie.
    Pouvez-vous confirmer votre disponibilité ?
    """

    analyse = agent_analyseur().analyse(email)
    emails_similaires = AgentRecuperateur().recupere(analyse, top_k=2)

    redacteur = agent_redacteur()
    reponse = redacteur.generer_reponse(analyse, emails_similaires)

    print("\nRéponse générée :\n")
    print(reponse)

    assert isinstance(reponse, str)
    assert len(reponse) > 0
    assert 'Bonjour' in reponse or 'Merci' in reponse  


if __name__ == "__main__":
    test_agent_redacteur_llm()
