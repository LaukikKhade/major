import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from models.unet import UNet


# 🔥 Dice Loss (strong but stable)
class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        smooth = 1e-6
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice


# 📦 DATASET (F + BC + CA ONLY)
class LocalizationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = [
            f for f in os.listdir(img_dir)
            if (
                f.endswith("_F.png") or
                "_BC" in f or
                "_CA" in f
            )
        ]

        self.images = sorted(self.images)

        # 🔥 AUGMENTATION (important)
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return img, mask


# ⚙️ LOAD DATA
dataset = LocalizationDataset(
    "data/localization/images",
    "data/localization/masks"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)


# ⚙️ DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🧠 MODEL
model = UNet().to(device)


# 🔥 BALANCED LOSS
pos_weight = torch.tensor([12.0]).to(device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 20


# 🚀 TRAINING LOOP
total_start = time.time()

for epoch in range(epochs):
    start_time = time.time()

    total_loss = 0
    total_correct = 0
    total_pixels = 0
    total_iou = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # 🔥 FINAL LOSS
        loss = bce(outputs, masks) + dice(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔍 PREDICTIONS
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.4).float()

        # Accuracy
        correct = (preds == masks).sum().item()
        total_correct += correct
        total_pixels += masks.numel()

        # IoU
        intersection = (preds * masks).sum().item()
        union = ((preds + masks) > 0).sum().item()

        if union > 0:
            total_iou += intersection / union

        loop.set_postfix(loss=loss.item())

    epoch_time = time.time() - start_time
    accuracy = total_correct / total_pixels
    avg_iou = total_iou / len(loader)

    print(f"\nEpoch {epoch+1} Completed")
    print(f"Loss: {total_loss/len(loader):.4f} | Accuracy: {accuracy:.4f} | IoU: {avg_iou:.4f}")
    print(f"Time per epoch: {epoch_time:.2f} sec")

    torch.save(model.state_dict(), "unet.pth")


total_time = time.time() - total_start
print(f"\nTotal training time: {total_time/60:.2f} minutes")