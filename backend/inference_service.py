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

# Limit CPU threads to 4 to prevent 100% CPU starvation and system freezing
torch.set_num_threads(4)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Configuration ────────────────────────────────────────────────────────────
CLEAN_MODEL_PATH = os.path.join(PROJECT_ROOT, "convnextv2_large_clean.pth")
EPOCH_MODEL_PATH = os.path.join(PROJECT_ROOT, "convnextv2_large_epoch_25_ema.pth")
MODEL_PATH = CLEAN_MODEL_PATH if os.path.exists(CLEAN_MODEL_PATH) else EPOCH_MODEL_PATH
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# ─── Calibrated Clinical Thresholds ───────────────────────────────────────────
OPTIMIZED_THRESHOLDS = [0.6925, 1.6520, 1.9061, 3.2191]

def build_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def detect_eye_side_from_image(img_rgb: np.ndarray) -> str:
    """
    Anatomically detects Left Eye (OS) vs Right Eye (OD) based on optic disc position.
    """
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(thresh) == 0:
            _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(thresh) == 0:
            _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
            
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                w = img_rgb.shape[1]
                if cx > w / 2:
                    return "right"
                else:
                    return "left"
    except Exception as e:
        print(f"[AI] Eye detection failed: {e}. Defaulting to left.")
    return "left"


# ─── Authentic Training-Matched Grad-CAM ──────────────────────────────────────
class AuthenticGradCAM:
    """
    Authentic Grad-CAM exactly matching src/evaluation/gradcam.py:
    Attaches to the backbone's feature output, propagates through FRG + CBAM + Head,
    and maps the gradients of the continuous severity regression score back to spatial dimensions.
    """
    def __init__(self, model):
        self.model = model

    def generate(self, img_tensor, img_h, img_w):
        # 1. Forward pass through backbone in inference mode
        with torch.inference_mode():
            backbone_features = self.model.backbone(img_tensor)

        # 2. Enable gradient on the backbone feature map
        feat = backbone_features.detach().clone().requires_grad_(True)
        frg_weight = self.model.frg(feat)
        refined = self.model.cbam(feat, frg_weight=frg_weight)
        pooled = self.model.global_pool(refined).flatten(1)
        outputs = self.model.head(pooled)

        # 3. Compute CORN probabilities and continuous score
        logits = outputs['ordinal_logits']
        s = torch.sigmoid(logits[0])
        p0 = (1.0 - s[0]).item()
        p1 = (s[0] * (1.0 - s[1])).item()
        p2 = (s[0] * s[1] * (1.0 - s[2])).item()
        p3 = (s[0] * s[1] * s[2] * (1.0 - s[3])).item()
        p4 = (s[0] * s[1] * s[2] * s[3]).item()
        probs = [p0, p1, p2, p3, p4]

        expected_score = float(sum(k * p for k, p in enumerate(probs)))
        reg_score = float(outputs['regression_score'][0].item())

        # Clinical grade determination
        if expected_score < OPTIMIZED_THRESHOLDS[0]:
            final_grade = 0
        elif expected_score < OPTIMIZED_THRESHOLDS[1]:
            final_grade = 1
        elif expected_score < OPTIMIZED_THRESHOLDS[2]:
            final_grade = 2
        elif expected_score < OPTIMIZED_THRESHOLDS[3]:
            final_grade = 3
        else:
            final_grade = 4

        # Multi-class argmax refinement for clear extreme cases
        if p4 > 0.40 or (p4 > 0.30 and expected_score > 3.0):
            final_grade = 4
        elif p3 > 0.40 and expected_score > 2.0:
            final_grade = 3
        elif p0 > 0.70 and expected_score < 0.7:
            final_grade = 0

        # 4. Backpropagate with respect to continuous severity score
        self.model.head.zero_grad()
        if feat.grad is not None:
            feat.grad.zero_()
        outputs['regression_score'][0].backward()

        # 5. Spatial pooling of gradients (Exact formula from src/evaluation/gradcam.py)
        grads = feat.grad
        pooled_gradients = torch.mean(grads, dim=[0, 2, 3])
        activations = feat.detach()[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (img_w, img_h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        clamped_score = float(np.clip(expected_score, 0.0, 4.0))
        return heatmap_color, clamped_score, final_grade, probs


class DiabeticRetinopathyAI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        threads = 4 if self.device.type == 'cpu' else 1
        torch.set_num_threads(threads)
        print(f"[AI] Optimized PyTorch CPU inference threads: {threads}")
        
        self.model = None
        self._model_loading = False
        self.transform = build_transform()
        self._load_model()

    def _load_model(self):
        if self._model_loading or self.model is not None:
            return

        if not os.path.exists(MODEL_PATH):
            print(f"[WARNING] Model weights not found at: {MODEL_PATH}")
            self.model = None
            return

        self._model_loading = True
        try:
            from src.models.sota_dr_model import SOTA_DR_Model
            print(f"[AI] Loading SOTA_DR_Model (ConvNeXtV2-Large) from:\n  {MODEL_PATH}")
            model = SOTA_DR_Model(model_name='convnextv2_large', pretrained=False)
            ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False, mmap=True)
            state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            
            # Freeze all parameters for max speed
            for param in model.parameters():
                param.requires_grad = False
                
            print("[AI] Priming neural network execution kernels...")
            with torch.inference_mode():
                dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
                _ = model(dummy_input)
                
            self.model = model
            print("\n" + "="*80)
            print(f"  SUCCESS: SOTA DR Model loaded & primed on {self.device}")
            print("  Authentic Grad-CAM generation and rapid inference are ACTIVE!")
            print("="*80 + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback; traceback.print_exc()
            self.model = None
        finally:
            self._model_loading = False

    def preprocess_fundus(self, img_rgb: np.ndarray, desired_size: int = 1024) -> np.ndarray:
        """
        Exact Training Preprocessing:
        1. Resize directly to target square resolution using INTER_LANCZOS4.
        2. Light Y-Channel CLAHE (clipLimit=1.5, tileGridSize=(8,8)) in YUV space.
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (desired_size, desired_size), interpolation=cv2.INTER_LANCZOS4)
        
        img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_bgr_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2RGB)

    def _preprocess(self, img_np: np.ndarray):
        augmented = self.transform(image=img_np)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        return tensor

    def process_image(self, image_bytes: bytes, scan_id: int, filename: str = None):
        """
        Synchronous worker thread execution for non-blocking FastAPI background task.
        """
        print(f"[AI] Starting PyTorch inference for scan {scan_id} ...")

        # 1. Preprocess raw image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_raw = np.array(pil_img)
            img_np = self.preprocess_fundus(img_raw, desired_size=1024)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img)

        # 2. Overwrite saved raw image with clean preprocessed image
        UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        from database import SessionLocal
        from models import Scan

        db_session = SessionLocal()
        try:
            scan = db_session.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                raw_fname = os.path.basename(scan.raw_image_s3_url)
                raw_path = os.path.join(UPLOAD_DIR, raw_fname)
                cv2.imwrite(raw_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"[AI] Overwrote raw image for scan {scan_id} with preprocessed version.")
        except Exception as e:
            print(f"[ERROR] Failed to overwrite raw image: {e}")
        finally:
            db_session.close()

        gradcam_url = None
        regression_score = None
        dr_level = None

        if self.model is not None:
            try:
                img_tensor = self._preprocess(img_np)
                img_h, img_w = img_np.shape[:2]

                cam = AuthenticGradCAM(self.model)
                heatmap_bgr, regression_score, dr_level, probs = cam.generate(img_tensor, img_h, img_w)
                print(f"[AI Scan {scan_id}] Prediction -> Score: {regression_score:.4f} | DR Level: {dr_level} | Probs: {probs}")

                # Overlay using standard apply_heatmap (alpha=0.5) from training
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_bgr_resized = cv2.resize(img_bgr, (img_w, img_h))
                overlay = cv2.addWeighted(heatmap_bgr, 0.5, img_bgr_resized, 0.5, 0)

                # Save GradCAM image to uploads
                gradcam_filename = f"gradcam_{uuid.uuid4()}.jpg"
                gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)
                cv2.imwrite(gradcam_path, overlay)
                gradcam_url = f"{BASE_URL}/uploads/{gradcam_filename}"
                print(f"[AI Scan {scan_id}] GradCAM saved: {gradcam_filename}")

            except Exception as inf_err:
                print(f"[ERROR Scan {scan_id}] Inference failed: {inf_err}.")
                import traceback; traceback.print_exc()

        # 3. Update Database with true PyTorch results
        db_session = SessionLocal()
        try:
            scan = db_session.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                if scan.eye_side == "auto":
                    detected = detect_eye_side_from_image(img_np)
                    scan.eye_side = detected
                    print(f"[AI Scan {scan_id}] Auto-detected eye side: {detected}")
                scan.dr_prediction_level = dr_level
                scan.regression_score = regression_score
                scan.gradcam_image_s3_url = gradcam_url
                if dr_level is not None:
                    scan.status = "completed"
                else:
                    scan.status = "failed"
                db_session.commit()
                print(f"[AI Scan {scan_id}] Database updated successfully with status {scan.status} [OK]")
            else:
                print(f"[ERROR] Scan ID {scan_id} not found in database.")
        except Exception as e:
            print(f"[ERROR] Failed to update database: {e}")
            db_session.rollback()
        finally:
            db_session.close()

    def predict_quick(self, image_bytes: bytes, eye_side: str = "left", filename: str = None):
        """
        Database-agnostic fast live inference with real PyTorch model.
        """
        print(f"[AI Quick] Starting live inference...")
        
        UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        raw_filename = f"raw_{uuid.uuid4()}.jpg"
        gradcam_filename = f"gradcam_{uuid.uuid4()}.jpg"
        
        raw_path = os.path.join(UPLOAD_DIR, raw_filename)
        gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_raw = np.array(pil_img)
            img_np = self.preprocess_fundus(img_raw, desired_size=1024)
            if eye_side == "auto":
                eye_side = detect_eye_side_from_image(img_np)
        except Exception as e:
            print(f"[ERROR Quick] Preprocessing failed: {e}")
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img)

        cv2.imwrite(raw_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        raw_url = f"{BASE_URL}/uploads/{raw_filename}"
        gradcam_url = None
        regression_score = None
        dr_level = None
        probs = None

        if self.model is not None:
            try:
                img_tensor = self._preprocess(img_np)
                img_h, img_w = img_np.shape[:2]

                cam = AuthenticGradCAM(self.model)
                heatmap_bgr, regression_score, dr_level, probs = cam.generate(img_tensor, img_h, img_w)
                print(f"[AI Quick] Score: {regression_score:.4f} -> DR Level: {dr_level}")

                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_bgr_resized = cv2.resize(img_bgr, (img_w, img_h))
                overlay = cv2.addWeighted(heatmap_bgr, 0.5, img_bgr_resized, 0.5, 0)

                cv2.imwrite(gradcam_path, overlay)
                gradcam_url = f"{BASE_URL}/uploads/{gradcam_filename}"
                print(f"[AI Quick] GradCAM saved: {gradcam_filename}")

            except Exception as inf_err:
                print(f"[ERROR Quick] Inference failed: {inf_err}")
                import traceback; traceback.print_exc()

        return {
            "regression_score": regression_score,
            "dr_prediction_level": dr_level,
            "probabilities": probs,
            "raw_image_url": raw_url,
            "gradcam_image_url": gradcam_url,
            "eye_side": eye_side,
            "status": "completed" if dr_level is not None else "failed"
        }


AI_Agent = DiabeticRetinopathyAI()
