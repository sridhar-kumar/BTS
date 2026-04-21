import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import os

DEVICE = "cpu"


class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            in_channels=1,
            classes=1,
            encoder_weights=None
        )

    def forward(self, x):
        return torch.sigmoid(self.unet(x))


def load_trained_model(version="v1"):
    """
    Version-aware model loader
    Supports future model upgrades
    """

    model = UNetModel().to(DEVICE)

    # -------- Model Versioning --------
    if version == "v2":
        model_path = "best_model_v2.pt"
    else:
        model_path = "best_model.pt"

    # Safety check
    if not os.path.exists(model_path):
        model_path = "best_model.pt"

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model