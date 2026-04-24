#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models as tv_models


MODEL_BUILDERS = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
}

NESTED_STATE_KEYS = ("state_dict", "model_state_dict", "model", "net", "encoder", "backbone")
WRAPPER_PREFIXES = ("module.", "model.", "net.")
ENCODER_PREFIXES = ("encoder_q.", "encoder.", "backbone.")


def load_checkpoint(ckpt_path: Path) -> Any:
    try:
        return torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for nested_key in NESTED_STATE_KEYS:
            nested = ckpt.get(nested_key)
            if isinstance(nested, dict):
                try:
                    return extract_state_dict(nested)
                except ValueError:
                    pass

        tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if tensor_items:
            return tensor_items
    raise ValueError("Checkpoint does not contain a usable state dict.")


def strip_prefixes_repeatedly(key: str, prefixes: tuple[str, ...]) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def normalize_key(key: str) -> str:
    key = strip_prefixes_repeatedly(key, WRAPPER_PREFIXES)
    key = strip_prefixes_repeatedly(key, ENCODER_PREFIXES)
    return key


def build_reference_shape_maps(base_encoder: str) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    constructor = MODEL_BUILDERS[base_encoder]
    try:
        full_model = constructor(weights=None)
    except TypeError:
        full_model = constructor(pretrained=False)
    backbone = nn.Sequential(*list(full_model.children())[:-1], nn.Flatten())

    full_shapes = {k: tuple(v.shape) for k, v in full_model.state_dict().items()}
    backbone_shapes = {k: tuple(v.shape) for k, v in backbone.state_dict().items()}
    return full_shapes, backbone_shapes


def canonicalize_state_dict(raw_state: dict[str, torch.Tensor], base_encoder: str) -> tuple[dict[str, torch.Tensor], int]:
    full_shapes, backbone_shapes = build_reference_shape_maps(base_encoder)
    canonical: dict[str, torch.Tensor] = {}
    skipped = 0

    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = normalize_key(raw_key)
        if not normalized_key:
            skipped += 1
            continue

        tensor_shape = tuple(value.shape)
        full_shape = full_shapes.get(normalized_key)
        backbone_shape = backbone_shapes.get(normalized_key)
        if full_shape != tensor_shape and backbone_shape != tensor_shape:
            skipped += 1
            continue

        canonical[f"encoder_q.{normalized_key}"] = value

    if not canonical:
        raise ValueError(
            "No compatible backbone tensors were found after normalization. "
            "Check that base_encoder matches the checkpoint architecture."
        )
    return canonical, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a teammate/in-house pretrain checkpoint into a CACo-compatible "
            "state_dict format without modifying caco/ source code."
        )
    )
    parser.add_argument("--input_ckpt", type=Path, required=True)
    parser.add_argument("--output_ckpt", type=Path, required=True)
    parser.add_argument("--base_encoder", choices=sorted(MODEL_BUILDERS.keys()), default="resnet18")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_ckpt.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input_ckpt}")
    if args.output_ckpt.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {args.output_ckpt}. "
            "Use --overwrite to replace it."
        )

    checkpoint = load_checkpoint(args.input_ckpt)
    raw_state = extract_state_dict(checkpoint)
    canonical_state, skipped = canonicalize_state_dict(raw_state, args.base_encoder)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": canonical_state}, args.output_ckpt)

    print(
        f"[ok] wrote normalized checkpoint to {args.output_ckpt} "
        f"(kept={len(canonical_state)}, skipped={skipped})"
    )


if __name__ == "__main__":
    main()
