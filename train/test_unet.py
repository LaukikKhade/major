import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = UNet().to(device)
model.load_state_dict(torch.load("unet (1).pth", map_location=device))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------- INPUT IMAGE ----------------
img_path = "data/Classification/splicing/Tp_D_NRN_S_N_cha10108_cha10110_11585.jpg"  # change this
image = Image.open(img_path).convert("RGB")

img = transform(image).unsqueeze(0).to(device)

# ---------------- PREDICT MASK ----------------
with torch.no_grad():
    output = model(img)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()

# normalize for display
mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

# ---------------- SHOW RESULTS ----------------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image.resize((256,256)))
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.show()