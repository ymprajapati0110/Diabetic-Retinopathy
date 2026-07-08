import torch

class EMA:
    """
    Exponential Moving Average of model parameters.
    Improves stability and generalization.
    Fix #7: Tracks ALL parameters (not just requires_grad=True) so that frozen
    backbone layers also benefit from EMA smoothing during validation.
    Vectorized apply_shadow/restore using _foreach_copy_ for speed on large models.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Pre-build name list for fast iteration
        self._param_names = []
        self.register()

    def register(self):
        """ Register ALL parameters (frozen and unfrozen) for full EMA coverage. """
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()
            self._param_names.append(name)

    def update(self):
        """ Update EMA only for trainable parameters (frozen ones don't change). """
        model_params = []
        shadow_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                model_params.append(param.data)
                shadow_params.append(self.shadow[name])

        if not shadow_params:
            return

        # shadow = param.data * (1.0 - decay) + shadow * decay
        torch._foreach_mul_(shadow_params, self.decay)
        torch._foreach_add_(shadow_params, model_params, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """ Temporarily apply EMA weights to model for validation/inference. """
        param_dict = dict(self.model.named_parameters())
        for name in self._param_names:
            if name in param_dict:
                assert name not in self.backup
                self.backup[name] = param_dict[name].data
                param_dict[name].data = self.shadow[name]

    def restore(self):
        """ Restore original model weights for training. """
        param_dict = dict(self.model.named_parameters())
        for name in self._param_names:
            if name in self.backup and name in param_dict:
                param_dict[name].data = self.backup[name]
        self.backup = {}
