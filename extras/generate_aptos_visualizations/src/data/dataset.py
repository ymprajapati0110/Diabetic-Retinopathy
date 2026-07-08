import os
import cv2
cv2.setNumThreads(2)  # 2 threads/worker × 10 workers = 20 total, leaves headroom for main process
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_fold_augmentations(fold_num):
    """
    Returns fold-specific Albumentations transforms to maximize ensemble diversity.
    MixUp and CutMix are applied at the batch level during training, but here we 
    provide the structural/spatial augmentations tailored per fold.
    Note: target_size removed — images are already pre-resized in preprocessed_*/
    """
    # Base transforms: images already pre-sized
    base_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    
    # Fold-specific heavy transforms
    if fold_num == 1:
        specific_transforms = [A.RandomRotate90(p=0.5)]
    elif fold_num == 2:
        specific_transforms = [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)]
    elif fold_num == 3:
        # Fold 3: Lightened (Removed slow Elastic/Grid/Optical transforms)
        specific_transforms = [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.RandomBrightnessContrast(p=0.5),
        ]
    elif fold_num == 4:
        # Fold 4: Photometric heavy
        specific_transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.5),
            A.RandomGamma(p=0.5),
            A.Blur(blur_limit=3, p=0.2),
        ]
    else:
        # Fold 5: Balanced regularized
        specific_transforms = [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        ]
        
    final_transforms = base_transforms + specific_transforms + [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return A.Compose(final_transforms)

def get_validation_augmentations(target_size=None):
    """Clean pipeline for validation. target_size kept for API compatibility but unused."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class DRDataset(Dataset):
    """
    Diabetic Retinopathy Dataset loader.
    Images must already be pre-resized (preprocessed_512/, _768/, _1024/).
    """
    def __init__(self, df, img_dir, fold_num=None, is_train=True, img_col='image', label_col='level'):
        self.df = df
        self.img_dir = img_dir
        self.img_col = img_col
        self.label_col = label_col
        self.is_train = is_train
        
        # Setup albumentations — no target_size needed, images already pre-sized
        if self.is_train and fold_num is not None:
            self.transform = get_fold_augmentations(fold_num)
        else:
            self.transform = get_validation_augmentations()
        
        # PRE-BUILD PATH LOOKUP TABLE — eliminates 12 os.path.exists() calls per image
        possible_ext = ['.jpg', '.jpeg', '.png', '.JPG']
        search_dirs = [img_dir, os.path.join(img_dir, 'train'), os.path.join(img_dir, 'test')]
        existing_dirs = [d for d in search_dirs if os.path.exists(d)]
        
        self._path_cache = {}
        for img_id in df[img_col].unique():
            img_id_str = str(img_id).strip()
            img_id_base = img_id_str.split('.')[0] if '.' in img_id_str else img_id_str
            for d in existing_dirs:
                for ext in possible_ext:
                    candidate = os.path.join(d, img_id_base + ext)
                    if os.path.exists(candidate):
                        self._path_cache[img_id_base] = candidate
                        break
                if img_id_base in self._path_cache:
                    break

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Robust extension handling
        img_name = str(row[self.img_col]).strip()
        img_id = img_name.split('.')[0] if '.' in img_name else img_name
        
        # Fast O(1) lookup from pre-built cache
        img_path = self._path_cache.get(img_id)
        
        if img_path is None:
            raise RuntimeError(
                f"Image ID {img_id} not found in cache for {self.img_dir}. "
                f"The image may have been filtered during preprocessing."
            )
        
        # Faster I/O using OpenCV
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img_bgr is None:
            raise RuntimeError(
                f"Failed to read image: {img_path}. "
                "Check that the preprocessed folder is complete and the CSV path is correct."
            )
        
        # OpenCV BGR -> RGB
        img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        # Albumentations (Augmentation & ToTensor)
        augmented = self.transform(image=img_np)
        img_tensor = augmented['image']
        
        # Targets
        label = row[self.label_col]
        target = {
            'ordinal': torch.tensor(label, dtype=torch.long),
            'binary': torch.tensor(1.0 if label >= 2 else 0.0, dtype=torch.float32),
            'regression': torch.tensor(float(label), dtype=torch.float32)
        }
        
        return img_tensor, target

def mixup_collate(batch):
    return mixup_cutmix_collate(batch, mode='mixup')

def cutmix_collate(batch):
    return mixup_cutmix_collate(batch, mode='cutmix')

def mixup_cutmix_collate(batch, alpha_mixup=0.4, alpha_cutmix=1.0, mode='mixup'):
    """
    Custom collate function to perform MixUp or CutMix at the batch level.
    Used primarily for Folds 1 and 2.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    
    # Unpack targets
    labels_ordinal = torch.stack([t['ordinal'] for t in targets])
    labels_binary = torch.stack([t['binary'] for t in targets])
    labels_reg = torch.stack([t['regression'] for t in targets])
    
    batch_size = images.size(0)
    
    # If mode isn't specified, just return standard batch
    if mode not in ['mixup', 'cutmix']:
        return images, {'ordinal': labels_ordinal, 'binary': labels_binary, 'regression': labels_reg}

    # Generate random permutation for the batch
    indices = torch.randperm(batch_size)
    
    if mode == 'mixup':
        lam = np.random.beta(alpha_mixup, alpha_mixup)
        # Mix images
        images = lam * images + (1 - lam) * images[indices, :]
        # Mix regression and binary targets
        labels_binary = lam * labels_binary + (1 - lam) * labels_binary[indices]
        labels_reg = lam * labels_reg + (1 - lam) * labels_reg[indices]
        # Ordinal targets can't be strictly mixed; return both targets + lambda
        # for the mixed CORN loss computed in train_one_epoch.
        
    elif mode == 'cutmix':
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        # Generate bounding box
        H, W = images.shape[2], images.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        # Adjust lambda to exact box ratio area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-1] * images.shape[-2]))
        
        labels_binary = lam * labels_binary + (1 - lam) * labels_binary[indices]
        labels_reg = lam * labels_reg + (1 - lam) * labels_reg[indices]
        
    return images, {
        'ordinal': labels_ordinal, 
        'ordinal_perm': labels_ordinal[indices],
        'binary': labels_binary, 
        'regression': labels_reg,
        'lam': lam,
        'mix_mode': mode
    }
