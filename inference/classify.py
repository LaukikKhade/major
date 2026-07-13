import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder("data/Classification", transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("classifier.pth", map_location=device))
model = model.to(device)
model.eval()

classes = dataset.classes

indices = random.sample(range(len(dataset)), 5)

for idx in indices:
    img, label = dataset[idx]

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        _, pred = torch.max(output, 1)

    img_np = img.permute(1, 2, 0).numpy()

    plt.imshow(img_np)
    plt.title(f"Pred: {classes[pred.item()]} | True: {classes[label]}")
    plt.axis("off")
    plt.show()