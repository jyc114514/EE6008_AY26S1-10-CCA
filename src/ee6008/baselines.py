"""Small diagnostic baselines with no network-dependent weight loading."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18


def load_r3d18(
    device: torch.device, *, weights: str = "none"
) -> torch.nn.Module:
    if weights == "none":
        model = r3d_18(weights=None)
    elif weights == "kinetics400_v1":
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        raise ValueError(f"unsupported R3D-18 weights: {weights}")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_r3d18(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    """Return the 512-D penultimate feature for an ImageNet-normalized clip."""
    if clip.ndim != 4:
        raise ValueError(f"expected C,T,H,W clip, got {tuple(clip.shape)}")
    # ``run_core_a`` already performs the shared RGB resize/crop and ImageNet
    # normalization. Undo only that normalization before applying the official
    # Kinetics-400 R3D normalization. Keep the temporal dimension unchanged.
    imagenet_mean = clip.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1, 1)
    imagenet_std = clip.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1, 1)
    x = (clip * imagenet_std + imagenet_mean).unsqueeze(0)
    x = F.interpolate(
        x,
        size=(x.shape[2], 112, 112),
        mode="trilinear",
        align_corners=False,
    )
    x = (
        x - x.new_tensor((0.43216, 0.394666, 0.37645)).view(1, 3, 1, 1, 1)
    ) / x.new_tensor((0.22803, 0.22145, 0.216989)).view(1, 3, 1, 1, 1)
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)[0]
