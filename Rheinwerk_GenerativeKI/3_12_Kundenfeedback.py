
#!/usr/bin/env python3
"""
Kundenfeedback-Analyse mit Sentiment-Analyse und Kategorisierung

Dieses Skript analysiert Kundenfeedback in verschiedenen Sprachen und 
klassifiziert es nach Sentiment und Kategorie.
"""

#%% Pakete
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Optional
import logging

#%% Konfiguration
# Logging-Einstellungen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modell-Konfigurationen
MODEL_CONFIG = {
    "zero_shot": "facebook/bart-large-mnli",
    "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment"
}

# Categories for classification
FEEDBACK_KATEGORIEN = ['defect', 'delivery', 'interface', 'customer_service', 'pricing']
 
#%% Daten
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus.",
    "The new update is amazing! The user interface is so much cleaner and faster. Great job!",
    "Terrible customer service. I've been waiting for 3 weeks for my refund and nobody responds to my emails."
]

# %% Klassen
class FeedbackAnalyzer:
    """
    Eine Klasse zur Analyse von Kundenfeedback mit Sentiment-Analyse und Kategorisierung.
    """
    
    def __init__(self, zero_shot_model: str = None, sentiment_model: str = None):
        """
        Initialisiert den FeedbackAnalyzer mit den angegebenen Modellen.
        
        Args:
            zero_shot_model: Modell für Zero-Shot-Klassifizierung
            sentiment_model: Modell für Sentiment-Analyse
        """
        self.zero_shot_model = zero_shot_model or MODEL_CONFIG["zero_shot"]
        self.sentiment_model = sentiment_model or MODEL_CONFIG["sentiment"]
        self._zero_shot_classifier = None
        self._sentiment_classifier = None
        
    def _initialize_classifiers(self):
        """Lazy Initialisierung der Klassifikatoren um unnötiges Laden von Modellen zu vermeiden."""
        if self._zero_shot_classifier is None:
            logger.info(f"Lade Zero-Shot-Klassifikator: {self.zero_shot_model}")
            self._zero_shot_classifier = pipeline(
                task="zero-shot-classification", 
                model=self.zero_shot_model
            )
        
        if self._sentiment_classifier is None:
            logger.info(f"Lade Sentiment-Klassifikator: {self.sentiment_model}")
            self._sentiment_classifier = pipeline(
                task="text-classification", 
                model=self.sentiment_model
            )
    
    def analyze_feedback(self, feedback_list: List[str], 
                        categories: List[str] = None) -> pd.DataFrame:
        """
        Analysiert eine Liste von Feedback-Texten für Sentiment und Kategorien.
        
        Args:
            feedback_list: Liste von Feedback-Strings zur Analyse
            categories: Liste von Kategorien für die Klassifizierung (Standard: FEEDBACK_KATEGORIEN)
            
        Returns:
            pd.DataFrame: DataFrame mit Feedback, Sentiment, Kategorie und Vertrauenswerten
        """
        if not feedback_list:
            logger.warning("Leere Feedback-Liste übergeben")
            return pd.DataFrame()
            
        categories = categories or FEEDBACK_KATEGORIEN
        self._initialize_classifiers()
        
        logger.info(f"Analysiere {len(feedback_list)} Feedback-Einträge")
        
        # Führe Zero-Shot-Klassifizierung durch
        zero_shot_results = self._zero_shot_classifier(
            feedback_list, 
            candidate_labels=categories
        )
        
        # Führe Sentiment-Analyse durch
        sentiment_results = self._sentiment_classifier(feedback_list)
        
        # Verarbeite Ergebnisse
        results = []
        for i, feedback_text in enumerate(feedback_list):
            # Behandle einzelne vs. Batch-Ergebnisse
            if isinstance(zero_shot_results, list):
                zs_result = zero_shot_results[i]
                sent_result = sentiment_results[i]
            else:
                zs_result = zero_shot_results
                sent_result = sentiment_results
            
            results.append({
                'feedback': feedback_text,
                'sentiment': sent_result['label'],
                'sentiment_score': sent_result['score'],
                'kategorie': zs_result['labels'][0],
                'kategorie_score': zs_result['scores'][0],
                'alle_kategorien': dict(zip(zs_result['labels'], zs_result['scores']))
            })
        
        return pd.DataFrame(results)

# %% Hilfsfunktionen
def print_analysis_summary(df: pd.DataFrame):
    """Gibt eine Zusammenfassung der Feedback-Analyse aus."""
    print("\n" + "="*60)
    print("FEEDBACK-ANALYSE ZUSAMMENFASSUNG")
    print("="*60)
    
    print(f"\nGesamtzahl Feedback-Einträge: {len(df)}")
    
    print("\nSentiment-Verteilung:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print("\nKategorie-Verteilung:")
    category_counts = df['kategorie'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\nDurchschnittliches Sentiment-Vertrauen: {df['sentiment_score'].mean():.3f}")
    print(f"Durchschnittliches Kategorie-Vertrauen: {df['kategorie_score'].mean():.3f}")
    
    # Zeige Beispiele für jede Kategorie
    print("\nBeispiel-Feedback nach Kategorie:")
    for category in df['kategorie'].unique():
        example = df[df['kategorie'] == category].iloc[0]
        print(f"\n{category.upper()}:")
        feedback_preview = example['feedback'][:100] + "..." if len(example['feedback']) > 100 else example['feedback']
        print(f"  \"{feedback_preview}\"")
        print(f"  Sentiment: {example['sentiment']} (Vertrauen: {example['sentiment_score']:.3f})")
        print(f"  Kategorie-Vertrauen: {example['kategorie_score']:.3f}")





# %% Hauptprogramm
def main():
    """Hauptfunktion, die die Feedback-Analyse-Fähigkeiten demonstriert."""
    print("Kundenfeedback-Analyse mit verbesserter Funktionalität")
    print("=" * 55)
    
    # Initialisiere Analyzer
    analyzer = FeedbackAnalyzer()
    
    # Analysiere Beispiel-Feedback
    logger.info("Analysiere Beispiel-Feedback...")
    results_df = analyzer.analyze_feedback(feedback)
    
    # Zeige detaillierte Zusammenfassung
    print_analysis_summary(results_df)
    
    # Zeige detaillierte Ergebnisse
    print("\n" + "="*60)
    print("DETAILLIERTE ERGEBNISSE")
    print("="*60)
    
    for i, row in results_df.iterrows():
        print(f"\nFeedback #{i+1}:")
        print(f"Text: {row['feedback'][:100]}{'...' if len(row['feedback']) > 100 else ''}")
        print(f"Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.3f})")
        print(f"Kategorie: {row['kategorie']} (Score: {row['kategorie_score']:.3f})")
        
        # Zeige Top 3 Kategorie-Vorhersagen
        top_categories = sorted(row['alle_kategorien'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top Kategorie-Vorhersagen:")
        for cat, score in top_categories:
            print(f"  - {cat}: {score:.3f}")


if __name__ == "__main__":
    main()

# %%
