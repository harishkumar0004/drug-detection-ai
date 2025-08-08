import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_data():
    with open("data/data_log.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    unique_data = []
    seen = set()
    for entry in data:
        identifier = (entry["timestamp"], entry["user"], entry["message"])
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(entry)
    
    os.makedirs("data", exist_ok=True)
    with open("data/data_log.json", "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=4, ensure_ascii=False)
        logger.info("Saved unique data to data/data_log.json")