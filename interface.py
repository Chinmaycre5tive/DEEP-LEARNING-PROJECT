import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# Load class names
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model definition (same as before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Inference function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# Create interface
gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs="label",
             title="CIFAR-10 Image Classifier",
             description="Upload an image to classify it among 10 CIFAR-10 classes."
).launch()
