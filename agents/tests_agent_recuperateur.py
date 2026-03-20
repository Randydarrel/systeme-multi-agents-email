

from agent_analyseur import agent_analyseur
from agent_recuperateur import AgentRecuperateur


def test_agent_recuperateur():
    email = """
    Une réunion est prévue la semaine prochaine concernant le trading de l'énergie.
    Pouvez-vous confirmer votre disponibilité ?
    """

    analyzer = agent_analyseur()
    analysis = analyzer.analyse(email)

    rag_agent = AgentRecuperateur()
    results = rag_agent.recupere(analysis, top_k=3)

    print("\nEmails récupérés :")

    for i, res in enumerate(results, 1):
        print(f"\n--- Email {i} ---")
        print("Distance :", res["distance"])
        print("Metadata :", res["metadata"])
        print("Contenu :", res["document"][:200])

    assert isinstance(results, list)

if __name__ == "__main__":
    test_agent_recuperateur()
