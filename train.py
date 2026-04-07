import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Dataset Load
train_dataset = datasets.ImageFolder(
    root="dataset/chest_xray/train",
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

# Simple CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*30*30,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Training
for epoch in range(3):
    running_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    print(f"Epoch {epoch+1}, Loss:{running_loss:.4f}")

print("Training Finished")