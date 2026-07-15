import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from models.unet import UNet

# ---------------- CONFIGURATION & DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your trained camera classes exactly as they were indexed during training
CAMERA_CLASSES = [
    "Agfa_DC-504_0",
    "Agfa_DC-733s",
    "Agfa_DC-830i_0",
    "Agfa_Sensor505-x_0",
    "Agfa_Sensor530s_0",
    "Canon_Ixus55_0",
    "Canon_Ixus70_0",
    "Canon_PowerShotA640",
    "Casio_EX_Z150_0",
    "FujiFilm_FinePixJ50_0",
    "Kodak_M1063_0",
    "Nikon_CoolPixS710_0",
    "Nikon_D200",
    "Nikon_D70",
    "Olympus_mju_1050SW",
    "Panasonic_DMC_FZ50",
    "Pentax_OptioA40",
    "Pentax_OptioW60",
    "Praktica_DCZ59",
    "Ricoh_GX100",
    "Samsung_L74wide",
    "Samsung_NV15",
    "Sony_DSC_H50",
    "Sony_DSC_T77",
    "Sony_DSC_W170",
    "rollei_RCP_7325XS"
] 
classes_tamper = ['AI', 'copy_move', 'real', 'splicing']

# ---------------- IMAGE TRANSFORMS ----------------
transform_cls = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_loc = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------- MODEL 1: Tampering Classifier (ResNet50) ----------------
cls_model = models.resnet50(pretrained=False)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 4)
cls_model.load_state_dict(torch.load("classifier_v2.pth", map_location=device))
cls_model = cls_model.to(device)
cls_model.eval()

# ---------------- MODEL 2: Tampering Localizer (UNet) ----------------
loc_model = UNet().to(device)
loc_model.load_state_dict(torch.load("unet.pth", map_location=device))
loc_model.eval()

# ---------------- MODEL 3: Source Camera Identifier (EfficientNet-B0) ----------------
camera_model_path = "camera_classifier.pth" 

camera_model = models.efficientnet_b0(pretrained=False)
in_features = camera_model.classifier[1].in_features
camera_model.classifier[1] = nn.Linear(in_features, len(CAMERA_CLASSES))
camera_model.load_state_dict(torch.load(camera_model_path, map_location=device))
camera_model = camera_model.to(device)
camera_model.eval()


# ---------------- EXECUTE PIPELINE ----------------
img_path = ""
image = Image.open(img_path).convert("RGB")

# Preprocess for classifiers
img_cls = transform_cls(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = cls_model(img_cls)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred = np.argmax(probs)

label = classes_tamper[pred]
print("Tamper Detection Prediction:", label)

# Secondary Check Variables
camera_label = None
camera_prob_text = ""

# If the image is authentic/real, trigger the Camera Identifier
if label == "real":
    print("-> Image verified as Authentic. Triggering Source Camera Identification...")
    with torch.no_grad():
        camera_output = camera_model(img_cls)
        camera_probs = torch.softmax(camera_output, dim=1).cpu().numpy()[0]
        camera_pred = np.argmax(camera_probs)
        
    camera_label = CAMERA_CLASSES[camera_pred]
    camera_prob_text = f"\nCamera: {camera_label} ({camera_probs[camera_pred]*100:.1f}%)"
    print(f"Identified Source: {camera_label}")


# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(14, 5))

# 1. INPUT IMAGE DISPLAY
plt.subplot(1, 3, 1)
plt.imshow(image.resize((224, 224)))
display_title = f"Input Image\nIntegrity: {label}"
if camera_label:
    display_title += camera_prob_text
plt.title(display_title)
plt.axis("off")

# 2. TAMPERING PROBABILITY BAR
plt.subplot(1, 3, 2)
plt.bar(classes_tamper, probs, color='skyblue')
plt.title("Integrity Class Probabilities")
plt.ylabel("Probability")
plt.ylim([0, 1])
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

# 3. CONTEXT-DEPENDENT THIRD PANEL (Mask or Status Text)
plt.subplot(1, 3, 3)
if label in ["copy_move", "splicing"]:
    img_loc = transform_loc(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = loc_model(img_loc)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()
    plt.imshow(mask, cmap="gray")
    plt.title("Tampering Mask")
    plt.axis("off")
elif label == "AI":
    plt.text(0.5, 0.5, "AI Generated Image\n(No Camera / No Mask)", fontsize=11, ha='center', va='center')
    plt.axis("off")
else: # label == "real"
    status_text = f"No Tampering Found\n\nVerified Source:\n{camera_label}"
    plt.text(0.5, 0.5, status_text, fontsize=12, ha='center', va='center', color='green', weight='bold')
    plt.axis("off")

plt.tight_layout()
plt.show()