#!/usr/bin/env python3
"""Export all final seed-3 validation outputs for the local numeric demo.

This exports archived outputs, never fresh model predictions. It requires only
NumPy, contains no raw transcripts/media/embeddings/weights, and permits explicit
input-path overrides. Run --help for the interface. Checkpoint hashing is opt-in
because the three checkpoints total about 5.2 GB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path("artifacts/demo")
DEFAULTS = {
    "mosi_readouts": BASE / "mosi_readouts.npz",
    "mosei_features": BASE / "mosei_readouts.npz",
    "iemocap_features": BASE / "iemocap_readouts.npz",
    "iemocap_metadata": BASE / "iemocap_metadata.json",
    "mosi_recovery_manifest": BASE / "mosi_recovery_manifest.json",
    "mosei_replay_audit": BASE / "mosei_replay_audit.json",
}
TRAIN_ROOT = {
    key: Path("artifacts/checkpoints") / key / "seed_3"
    for key in ("mosi", "mosei", "iemocap")
}
EPOCHS = {"mosi": 7, "mosei": 2, "iemocap": 20}
SIZES = {"mosi": 229, "mosei": 1871, "iemocap": 1031}
EPS = 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_record(path: Path, workspace: Path) -> dict:
    try:
        label = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        label = path.name
    return {"path": label, "sha256": sha(path), "bytes": path.stat().st_size}


def scalar(value) -> float:
    value = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("Nonfinite scalar in archive")
    return value


def vec(archive, key: str, index: int):
    if key not in archive.files:
        return None
    arr = np.asarray(archive[key][index])
    if not np.isfinite(arr).all():
        raise ValueError(f"Nonfinite {key}")
    return (
        arr.astype(bool).tolist()
        if arr.dtype.kind == "b"
        else arr.astype(float).tolist()
    )


def weighted_f1(y, pred, count=4) -> float:
    result = 0.0
    for label in range(count):
        tp = int(np.sum((y == label) & (pred == label)))
        fp = int(np.sum((y != label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        denom = 2 * tp + fp + fn
        result += float(np.mean(y == label)) * (2 * tp / denom if denom else 0.0)
    return result * 100


def recommended_ids(samples: list[dict], task: str) -> dict:
    result = {}
    for scenario in ("helpful", "kept", "harmful", "other"):
        items = [row for row in samples if row["scenario"] == scenario]
        if scenario == "helpful":
            items.sort(key=lambda row: row["error_gain"], reverse=True)
        elif scenario == "harmful":
            items.sort(key=lambda row: row["error_gain"])
        elif scenario == "kept":
            items.sort(key=lambda row: abs(row["error_gain"]))
        else:
            items.sort(key=lambda row: row["error_gain"], reverse=True)
        selected, seen = [], set()
        for row in items:
            label = row["y"] if task == "classification" else int(row["y"] >= 0)
            if label in seen:
                continue
            selected.append(row["id"])
            seen.add(label)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            selected.extend(row["id"] for row in items if row["id"] not in selected)
        result[scenario] = selected[:2]
    return result


def provenance(workspace: Path, key: str, cache: Path, verify_checkpoint: bool) -> dict:
    folder = workspace / TRAIN_ROOT[key]
    receipt_path = folder / "completed_training.json"
    validation_path = folder / "best_validation_predictions.npz"
    checkpoint_path = folder / "best.pt"
    receipt = json.loads(receipt_path.read_text())
    if receipt["best"]["epoch"] != EPOCHS[key]:
        raise ValueError(f"Unexpected selected epoch for {key}")
    expected = receipt["checkpoint_sha256"]
    actual = sha(checkpoint_path) if verify_checkpoint else None
    if actual is not None and actual != expected:
        raise ValueError(f"Checkpoint hash mismatch for {key}")
    return {
        "readout_cache": source_record(cache, workspace),
        "validation_cache": source_record(validation_path, workspace),
        "completed_training_receipt": source_record(receipt_path, workspace),
        "checkpoint": {
            "path": (TRAIN_ROOT[key] / "best.pt").as_posix(),
            "present": checkpoint_path.is_file(),
            "bytes": checkpoint_path.stat().st_size
            if checkpoint_path.exists()
            else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_verified": actual == expected if actual is not None else False,
            "included_in_demo_bundle": False,
        },
    }


def base_dataset(key: str, task: str, samples: list[dict], prov: dict) -> dict:
    if len(samples) != SIZES[key] or len({row["id"] for row in samples}) != SIZES[key]:
        raise ValueError(f"Incorrect or repeated IDs in {key}")
    return {
        "key": key,
        "name": key.upper(),
        "task": task,
        "split": "validation",
        "seed": 3,
        "epoch": EPOCHS[key],
        "n": SIZES[key],
        "mode": "archived_model_outputs",
        "modalities": ["text", "audio", "vision"],
        "provenance": prov,
        "statistics": {
            "scenario_counts": {
                name: sum(row["scenario"] == name for row in samples)
                for name in ["helpful", "kept", "harmful", "other"]
            }
        },
        "recommended_ids": recommended_ids(samples, task),
        "samples": samples,
    }


def export_scalar(
    key: str,
    cache: Path,
    workspace: Path,
    verify: bool,
    recovery_manifest: Path,
    replay_audit: Path,
) -> dict:
    data = np.load(cache, allow_pickle=False)
    val = np.load(
        workspace / TRAIN_ROOT[key] / "best_validation_predictions.npz",
        allow_pickle=False,
    )
    if not np.array_equal(data["ids"], val["ids"]) or not np.array_equal(
        data["y"], val["y"]
    ):
        raise ValueError(f"Validation identities/labels do not match for {key}")
    corrected_key = "corrected_prediction" if key == "mosi" else "logits"
    y = data["y"].reshape(-1).astype(float)
    text = data["text_prediction"].reshape(-1).astype(float)
    evidence = data["evidence_prediction"].reshape(-1).astype(float)
    corrected = data[corrected_key].reshape(-1).astype(float)
    recorded = val["logits"].reshape(-1).astype(float)
    max_corrected_delta = float(np.max(np.abs(corrected - recorded)))
    signs = data["q_sign"].reshape(-1).astype(float)
    magnitudes = data["q_magnitude"].reshape(-1).astype(float)
    if not (
        np.all((signs >= 0) & (signs <= 1))
        and np.all((magnitudes >= 0) & (magnitudes <= 1))
    ):
        raise ValueError("Gate outside [0,1]")
    samples, formula_deltas = [], []
    for index, sample_id in enumerate(data["ids"]):
        errors = {
            "text": abs(text[index] - y[index]),
            "evidence": abs(evidence[index] - y[index]),
            "corrected": abs(corrected[index] - y[index]),
        }
        gain = errors["text"] - errors["corrected"]
        scenario = "helpful" if gain > EPS else "harmful" if gain < -EPS else "kept"
        t = scalar(data["text_sign_logit"][index])
        f = scalar(data["fusion_sign_logit"][index])
        a_t = scalar(data["text_magnitude"][index])
        a_f = scalar(data["fused_magnitude"][index])
        residual = (abs(t) + 0.5) * np.tanh(f)
        manual = np.tanh(t + signs[index] * residual) * (
            (1 - magnitudes[index]) * a_t + magnitudes[index] * a_f
        )
        formula_deltas.append(abs(manual - corrected[index]))
        samples.append(
            {
                "id": str(sample_id),
                "y": float(y[index]),
                "split": "validation",
                "seed": 3,
                "epoch": EPOCHS[key],
                "text": float(text[index]),
                "evidence": float(evidence[index]),
                "corrected": float(corrected[index]),
                "q_sign": float(signs[index]),
                "q_magnitude": float(magnitudes[index]),
                "rho": vec(data, "rho", index),
                "u": vec(data, "u", index),
                "kappa": vec(data, "kappa", index),
                "available": vec(data, "available", index),
                "scenario": scenario,
                "absolute_error": errors,
                "error_gain": float(gain),
                "readout_state": {
                    "text_sign_logit": t,
                    "fusion_sign_logit": f,
                    "text_magnitude": a_t,
                    "fused_magnitude": a_f,
                    "beta": 0.5,
                },
            }
        )
    if max(formula_deltas) > 1e-4:
        raise ValueError(f"Dual-gate scalar equation does not match {key}")
    prov = provenance(workspace, key, cache, verify)
    if key == "mosei":
        extraction_metadata = cache.with_suffix(".json")
        extraction = json.loads(extraction_metadata.read_text())
        if extraction["checkpoint_sha256"] != prov["checkpoint"]["expected_sha256"]:
            raise ValueError("MOSEI feature-export checkpoint identity differs")
        old_replay = json.loads(replay_audit.read_text())
        archived_row = next(
            row for row in old_replay["rows"] if row["seed"] == 3 and row["epoch"] == 2
        )
        if abs(max_corrected_delta - archived_row["max_abs_error"]) > 1e-12:
            raise ValueError(
                "MOSEI replay drift does not match its previously recorded audit"
            )
        prov["feature_extraction_metadata"] = source_record(
            extraction_metadata, workspace
        )
        prov["existing_cpu_gpu_replay_audit"] = source_record(replay_audit, workspace)
    else:
        old_recovery = json.loads(recovery_manifest.read_text())["mosi_recovery"]
        if old_recovery["checkpoint_sha256"] != prov["checkpoint"]["expected_sha256"]:
            raise ValueError("MOSI recovered-head checkpoint identity differs")
        prov["existing_head_recovery_manifest"] = source_record(
            recovery_manifest, workspace
        )
    prov["replay_checks"] = {
        "ids_and_labels_match_saved_validation": True,
        "max_abs_corrected_difference_from_saved_validation": max_corrected_delta,
        "mean_abs_corrected_difference_from_saved_validation": float(
            np.mean(np.abs(corrected - recorded))
        ),
        "max_abs_dual_gate_formula_difference": max(formula_deltas),
        "saved_vs_exported_sign_decision_changes": int(
            np.sum((recorded >= 0) != (corrected >= 0))
        ),
    }
    if "recovered_corrected_prediction" in data.files:
        prov["replay_checks"]["max_abs_mosi_head_recovery_difference"] = float(
            np.max(
                np.abs(
                    data["recovered_corrected_prediction"]
                    - data["corrected_prediction"]
                )
            )
        )
    result = base_dataset(key, "sentiment", samples, prov)
    result["metric"] = "absolute_error"
    result["scenario_definitions"] = {
        "helpful": "Text absolute error minus corrected absolute error > 1e-6.",
        "harmful": "Text absolute error minus corrected absolute error < -1e-6.",
        "kept": "Absolute difference between those two errors is at most 1e-6.",
        "other": "Unused for finite scalar records.",
        "note": "These categories describe per-example regression error, not classification flips.",
    }
    for label, values in [
        ("text", text),
        ("evidence", evidence),
        ("corrected", corrected),
    ]:
        result["statistics"][label] = {
            "mae": float(np.mean(np.abs(values - y))),
            "acc2_including_zero_percent": float(
                np.mean((values >= 0) == (y >= 0)) * 100
            ),
        }
    result["statistics"]["mean_error_gain"] = float(
        np.mean(np.abs(text - y) - np.abs(corrected - y))
    )
    result["unrecorded_fields"] = [
        name for name in ["rho", "u", "kappa"] if name not in data.files
    ]
    result["notes"] = [
        "All validation observations are included. No per-sample text, media or feature embeddings are exported.",
        "The scalar output uses separate polarity and magnitude gates; it is not a linear interpolation of endpoint predictions.",
        "The evidence endpoint is tanh(f) times evidence magnitude; setting both scalar gates to one does not yield that endpoint.",
        "Feature/head replay outputs can differ numerically from the original GPU prediction cache; the exact maximum/mean and the prior replay audit are reported in provenance. They do not replace the paper's main results.",
    ]
    return result


def export_iemocap(cache: Path, metadata: Path, workspace: Path, verify: bool) -> dict:
    data = np.load(cache, allow_pickle=False)
    val = np.load(
        workspace / TRAIN_ROOT["iemocap"] / "best_validation_predictions.npz",
        allow_pickle=False,
    )
    meta = json.loads(metadata.read_text())
    classes = meta["class_names"]
    if classes != ["happy_or_excited", "sad", "neutral", "angry"]:
        raise ValueError("Unexpected actual IEMOCAP class ordering")
    if not np.array_equal(data["ids"], val["ids"]) or not np.array_equal(
        data["y"], val["y"]
    ):
        raise ValueError("IEMOCAP validation identities/labels differ")
    y = data["y"].astype(int)
    text = data["text_prob"].astype(float)
    evidence = data["fusion_prob"].astype(float)
    corrected = data["corrected_prob"].astype(float)
    q = data["q"].reshape(-1).astype(float)
    for name, values in [
        ("text", text),
        ("evidence", evidence),
        ("corrected", corrected),
    ]:
        if (
            values.shape != (1031, 4)
            or not np.isfinite(values).all()
            or not np.all(values >= 0)
        ):
            raise ValueError(f"Invalid {name} probability archive")
        if not np.allclose(values.sum(1), 1, atol=2e-6, rtol=0):
            raise ValueError(f"Unnormalized {name} probabilities")
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("Invalid contribution weight")
    mixture_delta = float(
        np.max(np.abs(corrected - ((1 - q[:, None]) * text + q[:, None] * evidence)))
    )
    if mixture_delta > 1e-6:
        raise ValueError("IEMOCAP correction is inconsistent with its actual mixture")
    tp, ep, cp = text.argmax(1), evidence.argmax(1), corrected.argmax(1)
    samples = []
    for index, sample_id in enumerate(data["ids"]):
        true = int(y[index])
        if tp[index] != true and cp[index] == true:
            scenario = "helpful"
        elif tp[index] == true and cp[index] == true:
            scenario = "kept"
        elif tp[index] == true and cp[index] != true:
            scenario = "harmful"
        else:
            scenario = "other"
        losses = {
            "text": float(-np.log(max(text[index, true], 1e-12))),
            "evidence": float(-np.log(max(evidence[index, true], 1e-12))),
            "corrected": float(-np.log(max(corrected[index, true], 1e-12))),
        }
        samples.append(
            {
                "id": str(sample_id),
                "y": true,
                "split": "validation",
                "seed": 3,
                "epoch": 20,
                "text_prob": text[index].tolist(),
                "evidence_prob": evidence[index].tolist(),
                "corrected_prob": corrected[index].tolist(),
                "q": float(q[index]),
                "rho": vec(data, "rho", index),
                "u": vec(data, "u", index),
                "kappa": vec(data, "kappa", index),
                "available": vec(data, "available", index),
                "prediction": {
                    "text": int(tp[index]),
                    "evidence": int(ep[index]),
                    "corrected": int(cp[index]),
                },
                "scenario": scenario,
                "nll": losses,
                "error_gain": losses["text"] - losses["corrected"],
            }
        )
    prov = provenance(workspace, "iemocap", cache, verify)
    extraction_metadata = cache.with_suffix(".json")
    extraction = json.loads(extraction_metadata.read_text())
    if extraction["checkpoint_sha256"] != prov["checkpoint"]["expected_sha256"]:
        raise ValueError("IEMOCAP feature-export checkpoint identity differs")
    prov["feature_extraction_metadata"] = source_record(extraction_metadata, workspace)
    prov["metadata"] = source_record(metadata, workspace)
    prov["replay_checks"] = {
        "ids_and_labels_match_saved_validation": True,
        "max_abs_mixture_formula_difference": mixture_delta,
        "max_abs_probability_difference_from_saved_validation": float(
            np.max(np.abs(corrected - np.exp(val["logits"].astype(float))))
        ),
        "saved_vs_exported_argmax_changes": int(np.sum(cp != val["logits"].argmax(1))),
    }
    result = base_dataset("iemocap", "classification", samples, prov)
    result["class_names"] = classes
    result["class_display_names"] = ["Happy / Excited", "Sad", "Neutral", "Angry"]
    result["metric"] = "negative_log_likelihood_nats"
    result["scenario_definitions"] = {
        "helpful": "Text argmax is wrong and corrected argmax is right.",
        "kept": "Both text and corrected argmax are right.",
        "harmful": "Text argmax is right and corrected argmax is wrong.",
        "other": "Both text and corrected argmax are wrong.",
        "note": "Classification correctness defines the scenarios; NLL gain is a separate numeric diagnostic.",
    }
    for label, values in [
        ("text", text),
        ("evidence", evidence),
        ("corrected", corrected),
    ]:
        pred = values.argmax(1)
        result["statistics"][label] = {
            "accuracy_percent": float(np.mean(pred == y) * 100),
            "weighted_f1_percent": weighted_f1(y, pred),
            "nll_nats": float(
                np.mean(-np.log(np.maximum(values[np.arange(len(y)), y], 1e-12)))
            ),
        }
    result["statistics"]["class_support"] = {
        name: int(np.sum(y == k)) for k, name in enumerate(classes)
    }
    result["notes"] = [
        "All 1,031 validation observations from selected seed-3 epoch-20 weights are included.",
        "These are uncalibrated archived readouts, not five-seed test-table metrics or real-time new-input inference.",
        "Source reliability rho and contribution gate q are different quantities.",
        "No raw transcripts, audio/video, original feature embeddings or weights are included.",
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Original experiment workspace root",
    )
    for name in DEFAULTS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            help="Override this source file; defaults to its registered relative path under --workspace",
        )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("data.json")
    )
    parser.add_argument(
        "--verify-checkpoints",
        action="store_true",
        help="Actually hash the three 1.74GB checkpoints against their locked receipts",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    paths = {
        name: getattr(args, name) or workspace / default
        for name, default in DEFAULTS.items()
    }
    datasets = {
        "mosi": export_scalar(
            "mosi",
            paths["mosi_readouts"],
            workspace,
            args.verify_checkpoints,
            paths["mosi_recovery_manifest"],
            paths["mosei_replay_audit"],
        ),
        "mosei": export_scalar(
            "mosei",
            paths["mosei_features"],
            workspace,
            args.verify_checkpoints,
            paths["mosi_recovery_manifest"],
            paths["mosei_replay_audit"],
        ),
        "iemocap": export_iemocap(
            paths["iemocap_features"],
            paths["iemocap_metadata"],
            workspace,
            args.verify_checkpoints,
        ),
    }
    if sum(item["n"] for item in datasets.values()) != 3131:
        raise ValueError("Incomplete validation cohorts")
    result = {
        "schema_version": 1,
        "mode": "archived_model_outputs",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "title": "TARC final-model validation readout replay",
        "total_records": 3131,
        "datasets": datasets,
        "scope": "Every selected seed-3 validation observation from all three final task models. No live inference is performed by the webpage.",
        "excluded": [
            "raw_transcripts",
            "raw_audio",
            "raw_video",
            "original_feature_embeddings",
            "checkpoint_weights",
        ],
        "interaction_scope": "Manually adjusted gates illustrate the recorded readout equation while holding its archived endpoints/components fixed. They are not freshly evaluated neural outputs.",
        "main_results_scope": "These single-seed validation diagnostics must not be presented as the five-seed test results in the paper.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": 3131,
                "scenario_counts": {
                    key: item["statistics"]["scenario_counts"]
                    for key, item in datasets.items()
                },
                "checkpoint_hashes_verified": args.verify_checkpoints,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
