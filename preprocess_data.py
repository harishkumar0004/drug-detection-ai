import json
import os
from datetime import datetime
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    # Load data
    with open("data/data_log.json", "r", encoding="utf-8") as f:
        data_log = json.load(f)
    
    static_image_dir = os.path.join("static", "telegram_images")
    os.makedirs(static_image_dir, exist_ok=True)
    
    for entry in data_log:
        # Clean text (remove URLs, normalize case)
        if entry.get("message"):
            entry["message"] = re.sub(r"http\S+", "", entry["message"]).strip().lower()
        
        entry["processed_timestamp"] = datetime.now().isoformat()
        entry["preprocessing_status"] = "completed"
        
        if entry["type"] == "photo" and entry.get("image_path"):
            filename = os.path.basename(entry["image_path"])
            original_image_path = os.path.join(static_image_dir, filename)
            
            if os.path.exists(original_image_path):
                # No need to copy — just fix the relative path
                entry["image_path"] = os.path.join("telegram_images", filename).replace("\\", "/")
                entry["image_processed"] = True
                logger.info(f"Image found: {original_image_path}")
            else:
                entry["image_processed"] = False
                logger.warning(f"Image not found: {original_image_path}")
        else:
            entry["image_processed"] = False
    
    with open("data/data_log.json", "w", encoding="utf-8") as f:
        json.dump(data_log, f, indent=4, ensure_ascii=False)
        logger.info("Saved preprocessed data to data/data_log.json")

if __name__ == "__main__":
    preprocess_data()
