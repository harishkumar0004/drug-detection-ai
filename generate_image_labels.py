import json
from datetime import datetime
from ultralytics import YOLO
import logging
import torch
from PIL import Image
import clip
import os

# -------------------
# CONFIG
# -------------------
CONF_THRESHOLD = 0.5
DRUG_RELATED_OBJECTS = {"bottle", "bag", "pill"}  # YOLO proxy objects
CLIP_PROMPTS = ["drug-related", "safe"]  # For zero-shot check
USE_CLIP = True  # Set False if you only want YOLO detection

# -------------------
# LOGGING
# -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# CLIP MODEL (optional)
# -------------------
if USE_CLIP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------
# MAIN FUNCTION
# -------------------
def generate_image_labels():
    # Load logs
    with open("data_log.json", "r", encoding="utf-8") as f:
        data_log = json.load(f)

    # Load or init labeled data
    if os.path.exists("data/labeled_data.json"):
        with open("data/labeled_data.json", "r", encoding="utf-8") as f:
            labeled_data = json.load(f)
    else:
        labeled_data = {"images": []}

    model = YOLO("yolov5s.pt")

    i = 0
    while i < len(data_log):
        text_entry = data_log[i] if (data_log[i]["type"] == "text" and "text_label" in data_log[i]) else None
        image_entry = data_log[i + 1] if (i + 1 < len(data_log) and data_log[i + 1]["type"] == "photo" and data_log[i + 1].get("image_processed", False)) else None

        # -------------------
        # Text-image pairing
        # -------------------
        if text_entry and image_entry:
            text_time = datetime.strptime(text_entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            image_time = datetime.strptime(image_entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            time_diff = (image_time - text_time).total_seconds()

            if 0 <= time_diff <= 5:
                inferred_label = 1 if text_entry["text_label"] == "drug-related" else 0
                labeled_data["images"].append({
                    "image_path": image_entry["image_path"],
                    "label": inferred_label,
                    "source": "text",
                    "confidence": 1.0
                })
                logger.info(f"[TEXT] {image_entry['image_path']} → Label {inferred_label}")
                i += 2
                continue

        # -------------------
        # YOLO detection
        # -------------------
        if i < len(data_log) and data_log[i]["type"] == "photo" and data_log[i].get("image_processed", False):
            image_path = data_log[i]["image_path"]

            # Skip duplicates
            if any(img["image_path"] == image_path for img in labeled_data["images"]):
                i += 1
                continue

            results = model(image_path)
            detections = []
            is_drug_yolo = False

            for pred in results:
                for box in pred.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    label_name = model.names[cls_id]
                    if conf >= CONF_THRESHOLD:
                        detections.append((label_name, conf))
                        if label_name in DRUG_RELATED_OBJECTS:
                            is_drug_yolo = True

            # -------------------
            # CLIP check (optional)
            # -------------------
            is_drug_clip = False
            clip_score = None
            if USE_CLIP:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                text_tokens = clip.tokenize(CLIP_PROMPTS).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    text_features = clip_model.encode_text(text_tokens)
                    logits_per_image, _ = clip_model(image_features, text_features)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                clip_score = float(probs[0])  # probability for "drug-related"
                is_drug_clip = clip_score > 0.5

            # -------------------
            # Final decision
            # -------------------
            final_label = 1 if (is_drug_yolo or is_drug_clip) else 0
            labeled_data["images"].append({
                "image_path": image_path,
                "label": final_label,
                "source": "yolo_clip" if USE_CLIP else "yolo",
                "detections": detections,
                "clip_score": clip_score
            })
            logger.info(f"[AUTO] {image_path} → Label {final_label} | Detections: {detections} | CLIP: {clip_score}")

        i += 1

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/labeled_data.json", "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=4, ensure_ascii=False)

    logger.info("✅ Auto-labeling complete.")

if __name__ == "__main__":
    generate_image_labels()
