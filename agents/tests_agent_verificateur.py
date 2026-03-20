from agent_analyseur import agent_analyseur
from agent_recuperateur import AgentRecuperateur
from agent_redacteur import agent_redacteur
from agent_verificateur import agent_verificateur


def test_agent_verificateur():
    email = """
    Une réunion est prévue la semaine prochaine concernant le trading de l'énergie.
    Pouvez-vous confirmer votre disponibilité ?
    """

    analyse = agent_analyseur().analyse(email)
    emails_similaires = AgentRecuperateur().recupere(analyse, top_k=2)

    redacteur = agent_redacteur()
    reponse_initiale = redacteur.generer_reponse(analyse, emails_similaires)

    verificateur = agent_verificateur()
    reponse_finale = verificateur.verifier_et_ameliorer(analyse, reponse_initiale)

    

    print("\nRéponse finale:\n")
    print(reponse_finale)

    assert isinstance(reponse_finale, str)
    assert len(reponse_finale) > 0


if __name__ == "__main__":
    test_agent_verificateur()
