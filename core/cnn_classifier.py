import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os

class CNNImageClassifier:
    def __init__(self, classes=None, model_path="models/cnn_model.pth"):
        if classes is None:
            self.classes = ["Hello", "iloveyou", "yes", "No", "Thankyou"]
        else:
            self.classes = classes
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data transforms adopted from the repository
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Setup lightweight ResNet18 model architecture
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        # Load weights if available
        full_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        self.is_trained = False
        if os.path.exists(full_model_path):
            try:
                self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))
                self.is_trained = True
                print(f"Loaded PyTorch CNN Classifier weights from {model_path}!")
            except Exception as e:
                print(f"Error loading PyTorch CNN weights: {e}")
        else:
            print("No CNN weights found. CNN Classifier will act as a dummy fallback until trained.")
            
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, base64_image):
        """
        Takes a base64 encoded image string, runs it through the PyTorch CNN,
        and returns (predicted_class, confidence_score).
        """
        if not self.is_trained:
            return "Unknown", 0.0
            
        try:
            # Decode image
            image_data = base64.b64decode(base64_image)
            img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Apply transforms
            img_tensor = self.data_transforms(img_pil).float()
            img_tensor = img_tensor.unsqueeze_(0)  # Add batch dimension
            
            inputs = img_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.classes[predicted_idx.item()]
                return predicted_class, confidence.item()
                
        except Exception as e:
            print(f"[CNN Classifier Error]: {e}")
            return "Unknown", 0.0
