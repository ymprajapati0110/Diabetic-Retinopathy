import os
import sys
import glob
import cv2
import torch
import numpy as np

torch.set_num_threads(min(8, os.cpu_count() or 4))
sys.path.insert(0, os.path.abspath(".."))
from src.models.sota_dr_model import SOTA_DR_Model
from src.data.preprocess import preprocess_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("[AI] Initializing ConvNeXt-V2 Large...", flush=True)
model = SOTA_DR_Model(model_name='convnextv2_large', pretrained=False)
ckpt = torch.load("../convnextv2_large_clean.pth", map_location='cpu')
state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# Let's find all success_gradcams test files
image_files = sorted(glob.glob("../extras/**/Grade*.jpg", recursive=True))
print(f"Total graded test images found: {len(image_files)}", flush=True)

for path in image_files:
    filename = os.path.basename(path)
    img = preprocess_image(path, 1024)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor)
        reg_score = out['regression_score'][0].item()
        
        # Ordinal sigmoid probabilities (CORN)
        s = torch.sigmoid(out['ordinal_logits'][0])
        p0 = (1.0 - s[0]).item()
        p1 = (s[0] * (1.0 - s[1])).item()
        p2 = (s[0] * s[1] * (1.0 - s[2])).item()
        p3 = (s[0] * s[1] * s[2] * (1.0 - s[3])).item()
        p4 = (s[0] * s[1] * s[2] * s[3]).item()
        
        probs = [p0, p1, p2, p3, p4]
        argmax_grade = int(np.argmax(probs))
        expected_score = float(sum(k * p for k, p in enumerate(probs)))
        
    print(f"File: {filename:<38} | ArgmaxGrade: {argmax_grade} | ExpScore: {expected_score:.2f} | Reg: {reg_score:+.2f} | P: [{p0:.2f}, {p1:.2f}, {p2:.2f}, {p3:.2f}, {p4:.2f}]", flush=True)
