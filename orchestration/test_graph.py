from orchestration.graph import build_graph

def test_graph():
    app = build_graph()

    email = """bonjour il y'a une réunion prevue la semaine prochaine"""

    result = app.invoke({"email": email})

    print("\n=== RÉPONSE FINALE ===\n")
    print(result['reponse_finale'])

if __name__ == "__main__":
    test_graph()