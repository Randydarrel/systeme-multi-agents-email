from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

def evaluer_reponse(reference,generation):
    # BLEU
    bleu = sentence_bleu([reference.split()], generation.split())

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, generation)

    # BERTScore
    P, R, F1 = score([generation], [reference], lang='fr')

    resultats = {
        "BLEU": bleu,
        "ROUGE-1": rouge['rouge1'].fmeasure,
        "ROUGE-2": rouge['rouge2'].fmeasure,
        "ROUGE-L": rouge['rougeL'].fmeasure,
        "BERTScore F1": F1.mean().item()
    }

    return resultats

reference = """
Une réunion est prévue la semaine prochaine concernant le trading de l'énergie.
    Pouvez-vous confirmer votre disponibilité ?
"""

generation = """
    Bonjour,

Je vous confirme ma disponibilité pour la réunion de la semaine prochaine consacrée au trading et à l'énergie, comme vous l'avez demandé. Je suis prêt à discuter des derniers développements et à échanger des idées avec l'équipe.

Cordialement,
"""

scores = evaluer_reponse(reference, generation)
print(scores)