import torch
import warnings
import inspect

def apply_patches():
    """
    Applies critical runtime patches. 
    Safe for both new PyTorch (2.6+) and older versions.
    """
    # 1. Capture original
    _original_torch_load = torch.load

    # 2. Check if the current torch version even supports 'weights_only'
    # (It was added in torch 1.13. If user has 1.12, passing it would crash)
    arg_spec = inspect.getfullargspec(_original_torch_load)
    supports_weights_only = 'weights_only' in arg_spec.args or 'weights_only' in arg_spec.kwonlyargs

    if supports_weights_only:
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False 
            return _original_torch_load(*args, **kwargs)
            
        torch.load = patched_load