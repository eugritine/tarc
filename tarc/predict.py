"""Command-line inference for final TARC checkpoints and processed NPZ inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Trusted complete final .pt checkpoint, not encoder initialization",
    )
    p.add_argument("--features", required=True, type=Path, help="Processed split .npz")
    p.add_argument("--dataset", choices=["mosi", "mosei", "iemocap"])
    p.add_argument("--config", type=Path, help="Override bundled dataset JSON")
    p.add_argument(
        "--encoder-config", type=Path, help="DeBERTa architecture config JSON only"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--indices", help="Comma-separated zero-based rows; default all samples"
    )
    p.add_argument(
        "--modalities", help="Optional comma-separated subset: text,audio,vision"
    )
    p.add_argument(
        "--output", type=Path, help="JSON destination; otherwise print to stdout"
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .api import load_model, predict_npz

    indices = [int(v) for v in args.indices.split(",")] if args.indices else None
    modalities = args.modalities.split(",") if args.modalities else None
    loaded = load_model(
        args.checkpoint, args.dataset, args.config, args.encoder_config, args.device
    )
    result = {
        "mode": "live_model_inference",
        "dataset": loaded.dataset,
        "checkpoint": loaded.metadata,
        "samples": predict_npz(
            loaded, args.features, indices, args.batch_size, modalities
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
