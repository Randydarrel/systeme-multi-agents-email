from orchestration.graph import build_graph


def test_pipeline_complet():

    graph = build_graph()

    email = """
    Bonjour,

    Pourrions-nous organiser une réunion la semaine prochaine
    afin de discuter du trading de l'énergie ?

    Cordialement,
    """

    
    result = graph.invoke({"email": email})

    
    assert isinstance(result, dict)
    assert "reponse_finale" in result
    assert isinstance(result["reponse_finale"], str)
    assert len(result["reponse_finale"]) > 50
