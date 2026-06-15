import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_cls = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_loc = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


cls_model = models.resnet50(pretrained=False)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 3)
cls_model.load_state_dict(torch.load("classifier.pth", map_location=device))
cls_model = cls_model.to(device)
cls_model.eval()

classes = ['copy_move', 'real', 'splicing']


loc_model = UNet().to(device)
loc_model.load_state_dict(torch.load("unet (1).pth", map_location=device))
loc_model.eval()


img_path = "data/Classification/copy_move/cmf_001_F_BC1.png"
image = Image.open(img_path).convert("RGB")

img_cls = transform_cls(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = cls_model(img_cls)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred = np.argmax(probs)

label = classes[pred]

print("Prediction:", label)


plt.figure(figsize=(12, 5))

# ---------------- IMAGE ----------------
plt.subplot(1, 3, 1)
plt.imshow(image.resize((224, 224)))
plt.title(f"Input Image\nPred: {label}")
plt.axis("off")


# ---------------- PROBABILITY BAR ----------------
plt.subplot(1, 3, 2)
plt.bar(classes, probs)
plt.title("Class Probabilities")
plt.ylabel("Probability")
plt.ylim([0, 1])

for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')


# ---------------- MASK (if tampered) ----------------
if label != "real":
    img_loc = transform_loc(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = loc_model(img_loc)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Tampering Mask")
    plt.axis("off")

else:
    plt.subplot(1, 3, 3)
    plt.text(0.2, 0.5, "No Tampering\n(Real Image)", fontsize=12)
    plt.axis("off")

plt.tight_layout()
plt.show()