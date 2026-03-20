from typing import TypedDict,List,Dict
from langgraph.graph import StateGraph,END
from agents.agent_analyseur import agent_analyseur
from agents.agent_recuperateur import AgentRecuperateur
from agents.agent_redacteur import agent_redacteur
from agents.agent_verificateur import agent_verificateur

class EmailState(TypedDict):
    email:str
    analyse:Dict
    emails_similaires:List[Dict]
    reponse_brouillon:str
    reponse_finale:str

analyseur = agent_analyseur()
recuperteur = AgentRecuperateur()
redacteur = agent_redacteur()
verificateur = agent_verificateur()

def analyse_node(state:EmailState)->EmailState:
    analyseur=agent_analyseur()
    state['analyse']=analyseur.analyse(state['email'])
    return state

def recuperation_node(state:EmailState)->EmailState:
    recuperateur=AgentRecuperateur()
    state['emails_similaires']=recuperateur.recupere(state['analyse'],top_k=3)
    return state

def redaction_node(state:EmailState)->EmailState:
    redacteur=agent_redacteur()
    state['reponse_brouillon']=redacteur.generer_reponse(state['analyse'],state['emails_similaires'])
    return state

def verification_node(state:EmailState)->EmailState:
    verificateur=agent_verificateur()
    state['reponse_finale']=verificateur.verifier_et_ameliorer(state['analyse'],state['reponse_brouillon'],)
    return state

def build_graph():
    graph = StateGraph(EmailState)

    graph.add_node("analyse", analyse_node)
    graph.add_node("recuperation", recuperation_node)
    graph.add_node("redaction", redaction_node)
    graph.add_node("verification", verification_node)

    graph.set_entry_point("analyse")

    graph.add_edge("analyse", "recuperation")
    graph.add_edge("recuperation", "redaction")
    graph.add_edge("redaction", "verification")
    graph.add_edge("verification", END)
    
    return graph.compile()