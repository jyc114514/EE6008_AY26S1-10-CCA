from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.videomae_encoder import (  # noqa: E402
    VideoMAEEncoder,
    load_videomae_config,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen VideoMAE smoke test.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "videomae_profile.yaml"),
        help="Path to the VideoMAE profile YAML.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model from the local Hugging Face cache only.",
    )
    return parser.parse_args()


def make_mock_video(
    shape: tuple[int, int, int, int, int],
    *,
    low: int,
    high: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(low, high + 1, shape, generator=generator, dtype=torch.uint8)


def main() -> int:
    args = parse_args()
    config = load_videomae_config(args.config)
    model_config = config["model"]
    smoke_config = config.get("smoke_test", {})
    device = select_device(force_cpu=args.cpu)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Current device: {device}")
    print(
        "GPU name: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Model name: {model_config['name']}")

    try:
        encoder = VideoMAEEncoder(
            model_config["name"],
            device=device,
            freeze=bool(model_config.get("freeze", True)),
            local_files_only=args.local_files_only,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        input_shape = encoder.expected_input_shape(
            batch_size=int(smoke_config.get("batch_size", 1))
        )
        videos = make_mock_video(
            input_shape,
            low=int(smoke_config.get("pixel_value_min", 0)),
            high=int(smoke_config.get("pixel_value_max", 255)),
            seed=int(smoke_config.get("seed", 6008)),
        )

        start = time.perf_counter()
        features = encoder(videos)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        counts = encoder.parameter_counts()
        has_nan = torch.isnan(features.pooled_features).any().item()
        has_inf = torch.isinf(features.pooled_features).any().item()
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )

        print(f"Input shape: {tuple(videos.shape)}")
        print(f"Token feature shape: {tuple(features.token_features.shape)}")
        print(f"Pooled feature shape: {tuple(features.pooled_features.shape)}")
        print(f"Total parameters: {counts['total']}")
        print(f"Frozen parameters: {counts['frozen']}")
        print(f"Trainable parameters: {counts['trainable']}")
        print(f"Contains NaN: {bool(has_nan)}")
        print(f"Contains Inf: {bool(has_inf)}")
        print(f"Inference time seconds: {elapsed:.4f}")
        print(f"Peak GPU memory MB: {peak_memory_mb:.2f}")

        passed = (
            encoder.is_frozen
            and features.pooled_features.ndim == 2
            and features.pooled_features.shape[0] == input_shape[0]
            and not has_nan
            and not has_inf
        )
        print(f"Smoke test {'passed' if passed else 'failed'}")
        return 0 if passed else 1
    except RuntimeError as error:
        print(f"Smoke test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
