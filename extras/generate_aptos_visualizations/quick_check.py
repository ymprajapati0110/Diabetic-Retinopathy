import os, torch, cv2, numpy as np

# 1. Check checkpoints exist
ckpt_dir = "d:/Major_project_shiro_final/results/fold_3"
for epoch in [23, 24, 25]:
    path = os.path.join(ckpt_dir, f"convnextv2_large_epoch_{epoch}_ema.pth")
    exists = os.path.exists(path)
    size_gb = os.path.getsize(path)/1e9 if exists else 0
    print(f"Epoch {epoch}: {'OK' if exists else 'MISSING'}  {size_gb:.2f} GB")

# 2. Check one test image preprocesses correctly
from src.data.preprocess import preprocess_image
test_img = "d:/Major_project_shiro_final/archive/resized test 15/10000_left.jpg"
img = preprocess_image(test_img, output_size=1024)
print(f"Preprocess output: {img.shape}, mean={img.mean():.1f}")  # expect (1024,1024,3), mean ~100-150

# 3. Check VRAM headroom at batch_size=4
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
