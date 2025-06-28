import os
import requests
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from PIL import Image

# Step 1: Download & Unzip the Dataset
FILE_ID = "PLEASE ADD THE DATASET LINK HERE"  # Replace with your actual Google Drive File ID
zip_url = f"https://drive.google.com/uc?id={FILE_ID}"
zip_path = "fabric_dataset.zip"
extract_path = "./data/dress_dataset"

# Download dataset if not present
if not os.path.exists(zip_path):
    print("📥 Downloading dataset...")
    response = requests.get(zip_url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("✅ Download complete!")

# Extract dataset if not already extracted
if not os.path.exists(extract_path):
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted!")
else:
    print("📂 Dataset already extracted.")

# Step 2: Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 3: Load dataset
train_dataset = ImageFolder(root=extract_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
fabric_labels = train_dataset.classes
print("🧵 Fabric pattern labels found:", fabric_labels)

# Step 4: Define the CNN model
class FabricCNN(nn.Module):
    def __init__(self, num_classes):
        super(FabricCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Step 5: Prepare for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FabricCNN(num_classes=len(fabric_labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📚 Epoch {epoch+1}: Loss = {running_loss:.4f}, Accuracy = {accuracy:.2f}%")

# Step 7: Classification Report
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=fabric_labels))

# Step 8: Visualize predictions
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images[:8]))
print("🧵 Predicted:", [fabric_labels[p] for p in preds[:8]])
print("🎯 Actual   :", [fabric_labels[l] for l in labels[:8]])

# Step 9: Save the model
torch.save(model.state_dict(), "pattern_sense_model.pth")
print("✅ Model saved as pattern_sense_model.pth")