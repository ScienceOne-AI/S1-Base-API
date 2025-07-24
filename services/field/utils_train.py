import torch



def load_model_weights(model, checkpoint_path):
    """
    Load model weights from a checkpoint, skipping incompatible layers.

    Args:
        model (torch.nn.Module): The PyTorch model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if '_orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]

    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if 'module.' in key:
            deal_key = key.replace('module.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]
    # Get state dictionaries
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Filter out mismatched weights
    filtered_state_dict = {}
    for name, param in checkpoint_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                filtered_state_dict[name] = param
            else:
                print(
                    f"Skipping weight: {name} due to shape mismatch ({param.shape} vs {model_state_dict[name].shape})")
        else:
            print(f"Skipping weight: {name} as it is not in the model")

    # Update model state dict with filtered weights
    model_state_dict.update(filtered_state_dict)

    # Load updated state dict into the model
    model.load_state_dict(model_state_dict)

    return model
