from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPModel:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased")

    def predict(self, text):
        try:
            result = self.classifier(text)[0]
            return {
                "label": result["label"],
                "confidence": result["score"]
            }
        except Exception as e:
            logger.error(f"NLP prediction failed: {e}")
            return {"label": "error", "confidence": 0}