import json
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset():
    # Load data
    with open("data/data_log.json", "r", encoding="utf-8") as f:
        data_log = json.load(f)
    
    # Prepare text data
    text_data = []
    for entry in data_log:
        if entry.get("message"):  # Only include entries with text
            text_data.append({
                "text": entry["message"],
                "label": 1 if entry.get("text_label") == "drug-related" else 0  # 1 for drug-related, 0 for safe
            })
    
    # Prepare image data
    image_data = []
    for entry in data_log:
        if entry["type"] == "photo" and entry.get("image_processed", False):
            image_path = os.path.join("data", entry["image_path"]).replace("/", os.sep)
            if os.path.exists(image_path):
                image_data.append({
                    "image_path": image_path,
                    "label": 1 if entry.get("image_label") == "drug-related" else 0  # 1 for drug-related, 0 for safe
                })
            else:
                logger.warning(f"Image not found: {image_path}")
    
    # Save to CSV
    text_df = pd.DataFrame(text_data)
    image_df = pd.DataFrame(image_data)
    
    os.makedirs("data/training", exist_ok=True)
    text_df.to_csv("data/training/text_data.csv", index=False)
    image_df.to_csv("data/training/image_data.csv", index=False)
    
    logger.info(f"Saved text data to data/training/text_data.csv ({len(text_data)} entries)")
    logger.info(f"Saved image data to data/training/image_data.csv ({len(image_data)} entries)")

if __name__ == "__main__":
    prepare_dataset()