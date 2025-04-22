import torch
import importlib

def get_vae_dtype():
    """
    Safely get the VAE dtype from model_management, with fallback to torch.float16
    """
    import comfy.model_management as model_management
    
    # Try to get VAE_DTYPES from model_management
    if hasattr(model_management, 'VAE_DTYPES'):
        return model_management.VAE_DTYPES[0]
    
    # Fallback: Try to get the default dtype used for VAE operations
    if hasattr(model_management, 'vae_dtype'):
        return model_management.vae_dtype
    elif hasattr(model_management, 'get_vae_dtype'):
        return model_management.get_vae_dtype()
    
    # If all else fails, use torch.float16 as a reasonable default
    return torch.float16 