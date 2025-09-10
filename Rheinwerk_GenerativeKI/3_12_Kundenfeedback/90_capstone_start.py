#%% packages
from transformers import pipeline
import pprint 

#%% data
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]

pprint.pprint(feedback)



       


# %% function
def process_zero_shot(feedback):
    classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
    
    #candidate_labels=["positive", "negative", "neutral"]
    candidate_labels=["defect", "delivery", "interface"]
    results = classifier(feedback, candidate_labels = candidate_labels)
    return results

#%% function for processing ratings
def process_ratings(feedback):
    sentiment_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment",return_all_scores=True)
    results = sentiment_model(feedback)
    return results

#%% Test
zero_shot = process_zero_shot(feedback)
pprint.pprint(zero_shot)


rating = process_ratings(feedback)
pprint.pprint(rating)

#%% print the zero shot classification with the highest probability per feedback
for i, text in enumerate(feedback):
    
    labels = zero_shot[i]['labels']
    scores = zero_shot[i]['scores']
    max_index = scores.index(max(scores))
    print(f"Feedback: {text}\nZero-Shot Classification: {labels[max_index]} (Score: {scores[max_index]})\n")

#%% print the rating with the highest propability per feedback and the highest zero shot classification
for i, text in enumerate(feedback):
    highest_rating = max(rating[i], key=lambda x: x['score'])
    print(f"Feedback: {text}\nPredicted Rating: {highest_rating['label']} (Score: {highest_rating['score']})")
    
    labels = zero_shot[i]['labels']
    scores = zero_shot[i]['scores']
    max_index = scores.index(max(scores))
    print(f"Zero-Shot Classification: {labels[max_index]} (Score: {scores[max_index]})\n")




# %%
