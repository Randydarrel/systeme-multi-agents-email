from orchestration.graph import build_graph


def test_coherence_reponse_metier():

    graph = build_graph()

    email = "Merci de confirmer votre disponibilité pour une réunion la semaine prochaine."

    result = graph.invoke({"email": email})

    reponse = result["reponse_finale"].lower()

    mots_cles = [
        "disponibilité",
        "réunion",
        "semaine",
        "créneau",
        "cordialement"
    ]

    assert any(mot in reponse for mot in mots_cles)
