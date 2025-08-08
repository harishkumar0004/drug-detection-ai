from models.nlp_model import NLPModel
from models.cv_model import CVModel
import json
from PIL import Image
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_labeled_data():
    with open("data/labeled_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["message"] for item in data["text"]]
    text_labels = [item["label"] for item in data["text"]]
    image_paths = [item["image_path"] for item in data["images"]]
    image_labels = [item["label"] for item in data["images"]]
    
    # Load images, skipping those that don't exist
    images = []
    valid_image_labels = []
    for path, label in zip(image_paths, image_labels):
        if os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')
                images.append(image)
                valid_image_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
        else:
            logger.warning(f"Image file not found: {path}. Skipping...")
    
    if not images:
        logger.error("No valid images found for training. Please check data/labeled_data.json and ensure image files exist.")
        raise ValueError("No valid images available for training.")
    
    return texts, text_labels, images, valid_image_labels

if __name__ == "__main__":
    texts, text_labels, images, image_labels = load_labeled_data()
    # Train text model
    logger.info("Training text model...")
    nlp_model = NLPModel()
    nlp_model.train(texts, text_labels)
    # Train image model
    logger.info("Training image model...")
    cv_model = CVModel()
    cv_model.train(images, image_labels)
    logger.info("Image model training complete")