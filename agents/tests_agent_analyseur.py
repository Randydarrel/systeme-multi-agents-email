

from agent_analyseur import agent_analyseur


def test_analyseur_agent():
    email = """
    Bonjour,

    Une réunion est prévue la semaine prochaine concernant le trading de l'énergie.
    Pouvez-vous confirmer votre disponibilité ?

    Cordialement
    """

    agent = agent_analyseur()
    result = agent.analyse(email)

    assert "intention" in result
    assert "ton" in result
    assert "questions" in result
    assert "entites" in result

    print("Résultat analyse :", result)


if __name__ == "__main__":
    test_analyseur_agent()
