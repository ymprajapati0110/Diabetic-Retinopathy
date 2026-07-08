import torch
import torch.nn.functional as F

class TTAWrapper:
    """
    Test-Time Augmentation (TTA).
    Applies 6 deterministic spatial transformations, runs them through the model, 
    and averages the predictions. Drastically reduces inference variance.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def _get_augmentations(self, image_batch):
        """ Generates 7 variants: Original, HFlip, VFlip, Rot90, Rot180, Rot270, Brightness+15% """
        # Brightness jitter: clamp keeps values in valid float range after normalization
        bright = torch.clamp(image_batch * 1.15, min=image_batch.min(), max=image_batch.max())
        return [
            image_batch,                                     # Original
            torch.flip(image_batch, dims=[3]),               # H-Flip
            torch.flip(image_batch, dims=[2]),               # V-Flip
            torch.rot90(image_batch, k=1, dims=[2, 3]),      # 90 deg
            torch.rot90(image_batch, k=2, dims=[2, 3]),      # 180 deg
            torch.rot90(image_batch, k=3, dims=[2, 3]),      # 270 deg
            bright,                                          # Brightness +15% jitter
        ]

    def __call__(self, x, temperature=1.5):
        """
        Runs the 6 variants, scales logits by Temperature to prevent
        over-confidence, and returns the averaged predictions.
        """
        augmentations = self._get_augmentations(x)
        
        all_ordinal_probs = []
        all_binary_probs = []
        all_regression_scores = []
        
        with torch.no_grad():
            for aug_img in augmentations:
                outputs = self.model(aug_img)
                
                # Temperature scaled probabilities for ordinal and binary
                ord_probs = torch.sigmoid(outputs['ordinal_logits'] / temperature)
                bin_probs = torch.sigmoid(outputs['binary_logits'] / temperature)
                
                all_ordinal_probs.append(ord_probs)
                all_binary_probs.append(bin_probs)
                all_regression_scores.append(outputs['regression_score'])
                
        # Average across the 6 TTA variants
        avg_ordinal = torch.mean(torch.stack(all_ordinal_probs), dim=0)
        avg_binary = torch.mean(torch.stack(all_binary_probs), dim=0)
        avg_regression = torch.mean(torch.stack(all_regression_scores), dim=0)
        
        return {
            'ordinal': avg_ordinal,
            'binary': avg_binary,
            'regression': avg_regression
        }

class SoftVotingEnsemble:
    """
    Combines N diverse models (10 in our architecture).
    Weighs their predictions based on their validation QWK score.
    """
    def __init__(self, models, weights=None):
        """
        models: List of loaded PyTorch models.
        weights: List of validation QWK scores for each model.
        """
        self.models = [TTAWrapper(m) for m in models]
        
        if weights is not None:
             # Normalize weights to sum to 1
             w = torch.tensor(weights, dtype=torch.float32)
             self.weights = w / w.sum()
        else:
             # Default to equal weighting
             self.weights = torch.ones(len(models)) / len(models)
             
    def forward(self, x, temperature=1.5):
        final_ordinal = 0
        final_binary = 0
        final_regression = 0
        
        for i, model in enumerate(self.models):
            w = self.weights[i]
            
            # TTA prediction from this specific model
            preds = model(x, temperature=temperature)
            
            final_ordinal += preds['ordinal'] * w
            final_binary += preds['binary'] * w
            final_regression += preds['regression'] * w
            
        return {
            'ensemble_ordinal': final_ordinal,
            'ensemble_binary': final_binary,
            'ensemble_regression': final_regression
        }
