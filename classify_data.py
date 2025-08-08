import json
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load text classifier
text_tokenizer = BertTokenizer.from_pretrained("models/text_classifier")
text_model = BertForSequenceClassification.from_pretrained("models/text_classifier")
text_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)

# Load image classifier
image_model = models.resnet18(weights=None)  # Updated from pretrained=False
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 2)
image_model.load_state_dict(torch.load("models/image_classifier.pth"))
image_model.eval()
image_model.to(device)

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_text(text):
    try:
        encoding = text_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
        
        label = "drug-related" if preds.item() == 1 else "safe"
        logger.info(f"Text classification: '{text}' -> {label}")
        return label
    except Exception as e:
        logger.error(f"Error classifying text '{text}': {e}")
        return "error"

def classify_image(image_path):
    try:
        image_path = os.path.join("data", image_path).replace("/", os.sep)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return "N/A"
        
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = image_model(image)
            preds = torch.argmax(outputs, dim=1)
        
        label = "drug-related" if preds.item() == 1 else "safe"
        logger.info(f"Image classification: {image_path} -> {label}")
        return label
    except Exception as e:
        logger.error(f"Error classifying image {image_path}: {e}")
        return "error"

def classify_data():
    # Load data
    with open("data/data_log.json", "r", encoding="utf-8") as f:
        data_log = json.load(f)
    
    for entry in data_log:
        # Text classification
        if entry.get("message"):
            entry["text_label"] = classify_text(entry["message"])
        else:
            entry["text_label"] = "N/A"
        
        # Image classification
        if entry["type"] == "photo" and entry.get("image_processed", False):
            entry["image_label"] = classify_image(entry["image_path"])
        else:
            entry["image_label"] = "N/A"
    
    # Save updated data
    with open("data/data_log.json", "w", encoding="utf-8") as f:
        json.dump(data_log, f, indent=4, ensure_ascii=False)
        logger.info("Saved classified data to data/data_log.json")

if __name__ == "__main__":
    classify_data()