from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models


MODEL_BUILDERS = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
}

WEIGHT_ENUMS = {
    "resnet18": getattr(tv_models, "ResNet18_Weights", None),
    "resnet34": getattr(tv_models, "ResNet34_Weights", None),
    "resnet50": getattr(tv_models, "ResNet50_Weights", None),
    "resnet101": getattr(tv_models, "ResNet101_Weights", None),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def default_learning_rate(train_mode: str) -> float:
    if train_mode == "finetune":
        return 1e-5
    return 1e-3


def _maybe_imagenet_weights(base_encoder: str):
    enum_cls = WEIGHT_ENUMS[base_encoder]
    if enum_cls is None:
        return None
    if hasattr(enum_cls, "DEFAULT"):
        return enum_cls.DEFAULT
    if hasattr(enum_cls, "IMAGENET1K_V1"):
        return enum_cls.IMAGENET1K_V1
    return None


def _build_resnet_backbone(base_encoder: str, backbone_type: str) -> Tuple[nn.Sequential, int]:
    constructor = MODEL_BUILDERS[base_encoder]
    if backbone_type == "imagenet":
        weights = _maybe_imagenet_weights(base_encoder)
        if weights is not None:
            model = constructor(weights=weights)
        else:
            model = constructor(pretrained=True)
    else:
        try:
            model = constructor(weights=None)
        except TypeError:
            model = constructor(pretrained=False)
    feature_dim = int(model.fc.in_features)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return encoder, feature_dim


def _extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if tensor_items:
            return tensor_items
    raise ValueError("Checkpoint does not contain a usable state dict.")


def _strip_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                out[key[len(prefix):]] = value
                break
    return out


def _load_if_matching(module: nn.Module, candidate_state: Dict[str, torch.Tensor]) -> int:
    module_keys = set(module.state_dict().keys())
    match_count = len(module_keys.intersection(candidate_state.keys()))
    if match_count == 0:
        return 0
    try:
        module.load_state_dict(candidate_state, strict=False)
    except RuntimeError:
        return 0
    return match_count


def _load_pretrained_into_encoder(
    encoder: nn.Sequential,
    base_encoder: str,
    ckpt_path: str,
) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = _extract_state_dict(ckpt)

    candidates = [
        raw_state,
        _strip_prefixes(raw_state, ("encoder_q.", "module.encoder_q.", "model.encoder_q.")),
        _strip_prefixes(raw_state, ("backbone.", "module.backbone.", "model.backbone.")),
    ]

    best_matches = 0
    for candidate in candidates:
        if not candidate:
            continue
        best_matches = max(best_matches, _load_if_matching(encoder, candidate))
        if best_matches > 50:
            return best_matches

    # Fallback: try loading as a full resnet and then re-create sequential encoder.
    full_model, _ = _build_resnet_backbone(base_encoder, backbone_type="random")
    resnet_model = MODEL_BUILDERS[base_encoder]()
    for candidate in candidates:
        if not candidate:
            continue
        matches = _load_if_matching(resnet_model, candidate)
        best_matches = max(best_matches, matches)
        if matches > 50:
            full_model = nn.Sequential(*list(resnet_model.children())[:-1], nn.Flatten())
            encoder.load_state_dict(full_model.state_dict(), strict=False)
            return best_matches

    return best_matches


def build_feature_extractor(
    backbone_type: str,
    base_encoder: str,
    ckpt_path: str | None = None,
) -> Tuple[nn.Sequential, int]:
    if base_encoder not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported base_encoder={base_encoder}")
    if backbone_type not in {"random", "imagenet", "pretrain"}:
        raise ValueError(f"Unsupported backbone_type={backbone_type}")

    if backbone_type in {"random", "imagenet"}:
        return _build_resnet_backbone(base_encoder, backbone_type)

    if ckpt_path is None:
        raise ValueError("ckpt_path is required when backbone_type='pretrain'.")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    encoder, feature_dim = _build_resnet_backbone(base_encoder, backbone_type="random")
    loaded = _load_pretrained_into_encoder(encoder, base_encoder, ckpt_path)
    if loaded == 0:
        raise ValueError(
            "Could not load encoder weights from checkpoint. "
            "Verify ckpt_path and base_encoder."
        )
    return encoder, feature_dim
