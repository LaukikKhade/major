import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(torch.load("unet.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img_dir = "data/localization/images"
mask_dir = "data/localization/masks"

files = sorted(os.listdir(img_dir))[:5]


for file in files:

    img_path = os.path.join(img_dir, file)
    mask_path = os.path.join(mask_dir, file)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().cpu().numpy()

    image = image.resize((256, 256))
    mask = mask.resize((256, 256))

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")

    plt.show()