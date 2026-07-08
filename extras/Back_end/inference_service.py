import asyncio
import os
import sys
import io
import uuid
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# ─── Add project root to sys.path so we can import src/ ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── Suppress noisy warnings ──────────────────────────────────────────────────
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, "convnextv2_large_epoch_25_ema.pth")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# ─── Preprocessing (mirrors validation pipeline from training) ────────────────
TARGET_SIZE = 1024  # Match the training resolution

def build_transform():
    return A.Compose([
        A.Resize(TARGET_SIZE, TARGET_SIZE),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ─── Thresholds (OptimizedRounder defaults, refined from training) ────────────
import json
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "gradcam_rank1_true3_pred-0.13", "fold_3_thresholds.json")
try:
    with open(THRESHOLDS_PATH, 'r') as f:
        data = json.load(f)
        DEFAULT_THRESHOLDS = data['thresholds']
except Exception as e:
    print(f"[WARNING] Could not load fold 3 thresholds, using default. Error: {e}")
    DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

def score_to_level(score: float, thresholds=DEFAULT_THRESHOLDS) -> int:
    for i, t in enumerate(thresholds):
        if score < t:
            return i
    return 4

# ─── GradCAM (adapted from src/evaluation/gradcam.py) ────────────────────────
class GradCAMHooker:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.backbone
        self._hook_fwd = self.target_layer.register_forward_hook(self._save_activation)
        self._hook_bwd = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._hook_fwd.remove()
        self._hook_bwd.remove()

    def generate(self, x_tensor, img_h, img_w):
        self.model.zero_grad()
        x_tensor.requires_grad_(True)
        outputs = self.model(x_tensor)
        score = outputs['regression_score'][0]
        score.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (img_w, img_h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_color, score.item()


class DiabeticRetinopathyAI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = build_transform()
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[WARNING] Model file not found at: {MODEL_PATH}")
            print("[WARNING] Running in MOCK mode. Place convnextv2_large_epoch_25_ema.pth in the DR1/ root folder.")
            self.model = None
            return

        try:
            from src.models.sota_dr_model import SOTA_DR_Model
            print(f"[AI] Loading SOTA_DR_Model (ConvNeXtV2-Large) from:\n  {MODEL_PATH}")
            model = SOTA_DR_Model(model_name='convnextv2_large', pretrained=False)
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            # Freeze model parameters to massively speed up GradCAM backward pass
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            print(f"[AI] Model loaded successfully on {self.device} ✓")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[WARNING] Falling back to MOCK mode.")
            self.model = None

    def _preprocess(self, image_bytes: bytes):
        """Convert raw image bytes → normalized tensor (1, 3, H, W)."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        augmented = self.transform(image=img_np)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        return tensor, img_np  # return original numpy for GradCAM overlay

    async def process_image(self, image_bytes: bytes, scan_id: int):
        print(f"[AI] Starting inference for scan {scan_id} ...")

        gradcam_url = None
        regression_score = None
        dr_level = None

        if self.model is not None:
            # ── Real inference ─────────────────────────────────────────────
            try:
                img_tensor, img_np = self._preprocess(image_bytes)
                img_h, img_w = img_np.shape[:2]

                # ── Inference & GradCAM (Single Pass) ────────────
                try:
                    hooker = GradCAMHooker(self.model)
                    heatmap_bgr, regression_score = hooker.generate(img_tensor, img_h, img_w)
                    hooker.remove_hooks()

                    dr_level = score_to_level(regression_score)
                    print(f"[AI] Score: {regression_score:.4f} → DR Level: {dr_level}")

                    # Overlay onto original image
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_bgr_resized = cv2.resize(img_bgr, (img_w, img_h))
                    overlay = cv2.addWeighted(heatmap_bgr, 0.45, img_bgr_resized, 0.55, 0)

                    # Save GradCAM image to uploads
                    UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    gradcam_filename = f"gradcam_{uuid.uuid4()}.jpg"
                    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)
                    cv2.imwrite(gradcam_path, overlay)
                    gradcam_url = f"{BASE_URL}/uploads/{gradcam_filename}"
                    print(f"[AI] GradCAM saved: {gradcam_filename}")

                except Exception as cam_err:
                    print(f"[WARNING] GradCAM failed: {cam_err}. Doing inference only.")
                    # Fallback to single forward pass without gradcam
                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        regression_score = outputs['regression_score'].item()
                    dr_level = score_to_level(regression_score)
                    print(f"[AI] Score: {regression_score:.4f} → DR Level: {dr_level}")
                    gradcam_url = None

            except Exception as inf_err:
                print(f"[ERROR] Inference failed: {inf_err}. Falling back to mock.")
                import traceback; traceback.print_exc()
                # Fall through to mock below
                regression_score = None
                dr_level = None

        # ── Update MySQL Database ──────────────────────────────────────────
        from database import SessionLocal
        from models import Scan

        db_session = SessionLocal()
        try:
            scan = db_session.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                scan.dr_prediction_level = dr_level
                scan.regression_score = regression_score
                scan.gradcam_image_s3_url = gradcam_url
                # Only set to completed if inference actually produced results
                if dr_level is not None:
                    scan.status = "completed"
                else:
                    scan.status = "failed"
                db_session.commit()
                print(f"[AI] MySQL updated for scan {scan_id} with status {scan.status} ✓")
            else:
                print(f"[ERROR] Scan ID {scan_id} not found in MySQL database.")
        except Exception as e:
            print(f"[ERROR] Failed to update MySQL: {e}")
            db_session.rollback()
        finally:
            db_session.close()


AI_Agent = DiabeticRetinopathyAI()
