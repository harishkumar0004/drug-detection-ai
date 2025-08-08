import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

class ImageClassifier:
    def __init__(self, model_path="models/resnet_drug.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ResNet50 model (matching cv_model.py)
        self.model = models.resnet50()  # Use ResNet50 instead of ResNet18
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        label = "drug-related" if predicted.item() == 1 else "safe"
        return label