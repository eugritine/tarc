"""Build a local encoder directory from a complete, trusted SST-2 initialization.

This converts tensor names and format without training. It does not produce the
SST-2 initialization itself or replace the original upstream backbone artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(initialization, config_path, output, expected_sha256, validate_only=False):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel

    initialization, config_path, output = map(
        Path, (initialization, config_path, output)
    )
    with initialization.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("Initialization SHA-256 mismatch")
    fields = json.loads(config_path.read_text())
    model_type = fields.pop("model_type")
    config = AutoConfig.for_model(model_type, **fields)
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(initialization, framework="pt", device="cpu") as source:
        keys = {
            key[len("bert.") :]: key for key in source.keys() if key.startswith("bert.")
        }
        shapes = {
            name: tuple(source.get_slice(key).get_shape()) for name, key in keys.items()
        }
        if shapes != expected:
            raise ValueError(
                "Initialization does not contain the complete configured encoder"
            )
        if validate_only:
            return {
                "status": "validated",
                "tensor_count": len(keys),
                "initialization_sha256": digest,
            }
        if output.exists():
            raise FileExistsError("Choose a new encoder output directory")
        output.mkdir(parents=True)
        try:
            tensors = {
                name: source.get_tensor(key).contiguous() for name, key in keys.items()
            }
            save_file(
                tensors, str(output / "model.safetensors"), metadata={"format": "pt"}
            )
            config.save_pretrained(output)
            receipt = {
                "source": "complete_shared_SST2_encoder_initialization",
                "initialization_sha256": digest,
                "tensor_count": len(keys),
                "training_updates": 0,
                "not_original_backbone_file": True,
            }
            (output / "conversion.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        except BaseException:
            shutil.rmtree(output)
            raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.initialization,
                args.config,
                args.output,
                args.expected_sha256,
                args.validate_only,
            )
        )
    )


if __name__ == "__main__":
    main()
