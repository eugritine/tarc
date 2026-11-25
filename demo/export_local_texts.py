#!/usr/bin/env python3
"""Match local original-release transcripts to the numeric demo's actual IDs.

Only use trusted local PKL releases: Python pickle is not an untrusted-data
format. This script loads no neural model, performs no inference, downloads
nothing, and exports no audio/video or feature embeddings. The resulting text
overlay is intended to stay outside the public repository and ZIP.

The overlay contains original release transcripts, not reconstructed sentences.
An original transcript may be longer than the visible window used by the model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path


DEFAULTS = {
    "mosi": Path("data/raw/aligned_50.pkl"),
    "mosei": Path("data/raw/mosei_aligned_50.pkl"),
    "iemocap": Path("data/raw/iemocap_dialoguecrn_features.pkl"),
}


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def string_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return None


def insert(mapping, conflicts, sample_id, text):
    sample_id, text = string_value(sample_id), string_value(text)
    if sample_id is None or text is None or not text.strip():
        return
    if sample_id in conflicts:
        return
    if sample_id in mapping and mapping[sample_id] != text:
        conflicts.add(sample_id)
        mapping.pop(sample_id)
        return
    mapping[sample_id] = text


def source_mapping(path: Path, dataset: str):
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    mapping, conflicts = {}, set()
    if dataset == "iemocap":
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            raise ValueError("Unexpected IEMOCAP feature-release structure")
        utterance_ids, transcripts = raw[0], raw[6]
        if not isinstance(utterance_ids, dict) or not isinstance(transcripts, dict):
            raise ValueError("IEMOCAP elements 0/6 must be dialogue dictionaries")
        for dialogue, ids in utterance_ids.items():
            texts = transcripts.get(dialogue)
            if texts is None or len(texts) != len(ids):
                raise ValueError(
                    f"Unpaired original transcript/ID arrays in {dialogue}"
                )
            for sample_id, text in zip(ids, texts):
                insert(mapping, conflicts, sample_id, text)
        scope = "Original DialogueCRN release utterance IDs (element 0) paired with original transcripts (element 6)."
    else:
        if not isinstance(raw, dict):
            raise ValueError("Expected an aligned-release split dictionary")
        split_name = "valid" if "valid" in raw else "val" if "val" in raw else None
        if split_name is None:
            raise ValueError("No original validation split is present")
        split = raw[split_name]
        ids = split.get("id", split.get("ids"))
        texts = split.get("raw_text")
        if ids is None or texts is None or len(ids) != len(texts):
            raise ValueError("Original validation IDs and raw_text do not pair")
        for sample_id, text in zip(ids, texts):
            insert(mapping, conflicts, sample_id, text)
        scope = (
            f"Original aligned release {split_name} split; id/ids paired with raw_text."
        )
    return mapping, sorted(conflicts), scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Existing trusted experiment workspace",
    )
    parser.add_argument(
        "--demo-data",
        type=Path,
        required=True,
        help="The numeric demo/data.json to match",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local text overlay outside the public repository",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Optional audit destination; defaults beside the overlay",
    )
    for key in DEFAULTS:
        parser.add_argument(
            f"--{key}-pkl", type=Path, help="Explicit trusted local source-PKL override"
        )
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("Place the local text overlay outside the public repository")
    demo = json.loads(args.demo_data.read_text())
    result = {
        "schema_version": 1,
        "scope": "local_original_release_transcripts",
        "note": "Original text is preserved exactly. It may exceed the model's visible word/subword window. No sentence is generated, paraphrased, reconstructed or padded.",
        "datasets": {},
    }
    audit = {
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "demo_data_sha256": sha(args.demo_data),
        "output_is_outside_public_repository": True,
        "raw_media_embeddings_or_weights_exported": False,
        "model_loaded": False,
        "model_inference": False,
        "training": False,
        "datasets": {},
    }
    for key, default in DEFAULTS.items():
        target = demo["datasets"][key]
        if target["split"] != "validation":
            raise ValueError(
                "This original-release export is scoped to validation examples"
            )
        wanted = [row["id"] for row in target["samples"]]
        if len(wanted) != len(set(wanted)):
            raise ValueError(f"Repeated demo IDs in {key}")
        path = getattr(args, f"{key}_pkl") or workspace / default
        record = {
            "source_path": str(path),
            "requested": len(wanted),
            "source_sha256": None,
            "matched": 0,
            "missing_ids": wanted,
            "conflicting_ids": [],
            "error": None,
        }
        found = {}
        try:
            mapping, conflicts, scope = source_mapping(path, key)
            found = {
                sample_id: mapping[sample_id]
                for sample_id in wanted
                if sample_id in mapping
            }
            record.update(
                {
                    "source_sha256": sha(path),
                    "source_bytes": path.stat().st_size,
                    "source_pairing": scope,
                    "matched": len(found),
                    "missing_ids": [
                        sample_id for sample_id in wanted if sample_id not in found
                    ],
                    "conflicting_ids": [
                        sample_id for sample_id in wanted if sample_id in conflicts
                    ],
                    "source_unique_transcripts": len(mapping),
                    "source_word_count_over_50": sum(
                        len(text.split()) > 50 for text in found.values()
                    ),
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            pickle.PickleError,
        ) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        result["datasets"][key] = found
        audit["datasets"][key] = record
    audit["total_requested"] = sum(
        row["requested"] for row in audit["datasets"].values()
    )
    audit["total_matched"] = sum(row["matched"] for row in audit["datasets"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    audit["output_path"] = str(output)
    audit["output_sha256"] = sha(output)
    audit["status"] = (
        "complete"
        if audit["total_matched"] == audit["total_requested"]
        else "partial_with_missing_ids_recorded"
    )
    audit_path = args.audit or output.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "matched": audit["total_matched"],
                "requested": audit["total_requested"],
                "status": audit["status"],
                "audit": str(audit_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
