import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the new weights API for torchvision
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)  # Binary classification: safe or drug-related
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"Initialized VisionModel on {self.device}")

    def preprocess(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            return image.to(self.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            return None

    def predict(self, image_path):
        image = self.preprocess(image_path)
        if image is None:
            return "N/A"

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        labels = ["safe", "drug-related"]
        return labels[predicted.item()]