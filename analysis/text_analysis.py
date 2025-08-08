import re
import logging
from typing import Dict, Any
import spacy
from textblob import TextBlob
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Perform comprehensive text analysis for drug-related content.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, Any]: Analysis results including entities, keywords, sentiment, etc.
    """
    try:
        # Basic text cleaning
        text = text.lower().strip()
        
        # SpaCy analysis
        doc = nlp(text)
        
        # Extract named entities
        entities = {ent.label_: ent.text for ent in doc.ents}
        
        # Extract key phrases (noun chunks)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Analyze sentiment using TextBlob
        blob = TextBlob(text)
        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # Extract potential drug-related keywords
        drug_keywords = _extract_drug_keywords(text)
        
        # Analyze message structure
        structure = {
            "word_count": len(doc),
            "sentence_count": len(list(doc.sents)),
            "avg_word_length": sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0,
            "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_phone_numbers": bool(re.search(r'\b\d{10,}\b', text))
        }
        
        # Get word frequencies
        word_freq = Counter([token.text for token in doc if not token.is_stop and not token.is_punct])
        
        # Calculate suspicion score
        suspicion_score = _calculate_suspicion_score(doc, drug_keywords, structure)
        
        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": sentiment,
            "structure": structure,
            "drug_keywords": drug_keywords,
            "frequent_words": dict(word_freq.most_common(10)),
            "suspicion_score": suspicion_score
        }
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        return {
            "error": str(e),
            "entities": {},
            "key_phrases": [],
            "sentiment": {"polarity": 0, "subjectivity": 0},
            "structure": {},
            "drug_keywords": [],
            "frequent_words": {},
            "suspicion_score": 0
        }

def _extract_drug_keywords(text: str) -> list:
    """
    Extract potential drug-related keywords from text.
    """
    # Common drug-related terms (example list, should be expanded)
    drug_terms = {
        'suspicious_amounts': r'\b\d+\s*(kg|g|mg|oz|pound|lb|gram|ounce)\b',
        'money_patterns': r'(\$\d+|\d+\s*(?:usd|eur|dollars|euros))',
        'time_patterns': r'\b\d{1,2}:\d{2}\b',
        'coded_language': r'\b(stuff|package|delivery|product|goods|special|quality)\b'
    }
    
    matches = {}
    for category, pattern in drug_terms.items():
        found = re.findall(pattern, text.lower())
        if found:
            matches[category] = found
            
    return matches

def _calculate_suspicion_score(doc, drug_keywords: dict, structure: dict) -> float:
    """
    Calculate a suspicion score based on various text features.
    """
    score = 0.0
    
    # Keywords presence
    score += len(drug_keywords) * 0.2
    
    # Structure indicators
    if structure["has_phone_numbers"]:
        score += 0.3
    if structure["has_urls"]:
        score += 0.2
        
    # Message characteristics
    if structure["word_count"] < 5:  # Very short messages might be suspicious
        score += 0.1
    
    # Normalize score to 0-1 range
    return min(1.0, score)

    def predict(self, image_path):
        self.model.eval()
        with torch.no_grad():
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            # Predict
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return "drug-related" if predicted.item() == 1 else "safe"

    def load_model(self, model_path="models/image_classifier.pth"):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()