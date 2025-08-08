import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image

class CVModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.labels = ["not-drug-related", "drug-related"]

    def load_model(self):
        # Load ResNet18 with updated weights parameter
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: drug-related, not-drug-related
        
        # Load pre-trained weights if available (you may need to train this model)
        # For now, we'll assume it's not fine-tuned
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                label = self.labels[predicted.item()]
            return label
        except Exception as e:
            raise Exception(f"Error predicting image: {str(e)}")

if __name__ == "__main__":
    # Example usage
    cv_model = CVModel()
    cv_model.load_model()
    label = cv_model.predict("path/to/image.jpg")
    print(f"Predicted label: {label}")