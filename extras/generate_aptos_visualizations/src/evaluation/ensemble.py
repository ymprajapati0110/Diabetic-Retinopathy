import torch
import torch.nn.functional as F

class TTAWrapper:
    """
    Test-Time Augmentation (TTA).
    Applies 6 deterministic spatial transformations, runs them through the model, 
    and averages the predictions. Drastically reduces inference variance.
    """
    def __init__(self, model, use_tta=True):
        self.model = model
        self.use_tta = use_tta
        self.model.eval()

    def _get_augmentations(self, image_batch):
        """5 deterministic geometric augmentations. No photometrics because they ruin regression QWK scale."""
        yield image_batch
        yield torch.flip(image_batch, dims=[3])  # H-flip
        yield torch.flip(image_batch, dims=[2])  # V-flip
        yield torch.rot90(image_batch, k=1, dims=[2, 3])  # Rotate 90
        yield torch.rot90(image_batch, k=3, dims=[2, 3])  # Rotate 270
    def __call__(self, x, temperature=1.5):
        """
        Runs the geometric variants natively.
        """
        if not self.use_tta:
            outputs = self.model(x)
            return {
                'ordinal': torch.sigmoid(outputs['ordinal_logits'] / temperature),
                'binary': torch.sigmoid(outputs['binary_logits'] / temperature),
                'regression': outputs['regression_score']
            }
            
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
    def __init__(self, models, weights=None, use_tta=True):
        """
        models: List of loaded PyTorch models.
        weights: List of validation QWK scores for each model.
        """
        self.models = [TTAWrapper(m, use_tta=use_tta) for m in models]
        
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
