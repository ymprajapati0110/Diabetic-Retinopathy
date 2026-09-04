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

THRESHOLDS = [0.6925, 1.6520, 1.9061, 3.2191]

def score_to_grade(score):
    return sum([score >= t for t in THRESHOLDS])

print("[AI] Initializing ConvNeXt-V2 Large...", flush=True)
model = SOTA_DR_Model(model_name='convnextv2_large', pretrained=False)
ckpt = torch.load("../convnextv2_large_clean.pth", map_location='cpu')
state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# Let's test specific grade files from success_gradcams
test_files = [
    "../extras/test_results/test_results/success_gradcams/Grade0_25365_left_conf91.0.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade0_29033_right_conf91.0.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade1_20343_right_conf69.8.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade1_31757_right_conf68.7.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade2_12059_right_conf81.8.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade2_19785_right_conf81.9.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade3_11578_right_conf68.0.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade3_13823_left_conf67.5.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade4_21311_right_conf95.0.jpg",
    "../extras/test_results/test_results/success_gradcams/Grade4_28720_left_conf94.9.jpg",
]

for path in test_files:
    filename = os.path.basename(path)
    if not os.path.exists(path):
        continue
    img = preprocess_image(path, 1024)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor)
        reg_score = out['regression_score'][0].item()
        
        # Ordinal sigmoid probabilities
        s = torch.sigmoid(out['ordinal_logits'][0])
        p0 = 1.0 - s[0].item()
        p1 = s[0].item() * (1.0 - s[1].item())
        p2 = s[0].item() * s[1].item() * (1.0 - s[2].item())
        p3 = s[0].item() * s[1].item() * s[2].item() * (1.0 - s[3].item())
        p4 = s[0].item() * s[1].item() * s[2].item() * s[3].item()
        corn_score = p0 * 0.0 + p1 * 1.0 + p2 * 2.0 + p3 * 3.0 + p4 * 4.0
        
        grade_reg = score_to_grade(reg_score)
        grade_corn = score_to_grade(corn_score)
        
    print(f"File: {filename:<38} | Reg: {reg_score:+.3f} (G{grade_reg}) | Corn: {corn_score:.3f} (G{grade_corn}) | p0:{p0:.2f} p1:{p1:.2f} p2:{p2:.2f} p3:{p3:.2f} p4:{p4:.2f}", flush=True)
