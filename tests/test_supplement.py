"""Asset integrity, encoder conversion and source snapshot regression checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SupplementTests(unittest.TestCase):
    def test_cpu_dependency_pins_are_unique(self):
        names = []
        for line in (
            (ROOT / "environment/requirements-cpu.lock.txt").read_text().splitlines()
        ):
            if not line or line.startswith(("#", "--")):
                continue
            name, version = line.split("==", 1)
            self.assertTrue(version)
            names.append(name.lower().replace("_", "-"))
        self.assertEqual(len(names), len(set(names)))

    def test_manifest_rejects_modified_release_file(self):
        verify = helper("verify_release").verify
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.py").write_text("original")
            digest = hashlib.sha256(b"original").hexdigest()
            (root / "MANIFEST.sha256").write_text(digest + "  source.py\n")
            self.assertEqual(verify(root), 1)
            (root / "source.py").write_text("changed")
            with self.assertRaises(ValueError):
                verify(root)

    def test_fifteen_runs_and_three_hundred_history_epochs(self):
        manifest = json.loads((ROOT / "assets/experiment_manifest.json").read_text())
        history = json.loads((ROOT / "results/training_histories.json").read_text())
        results = json.loads((ROOT / "results/final_results.json").read_text())
        total = 0
        for dataset in ("mosi", "mosei", "iemocap"):
            runs = manifest["datasets"][dataset]["runs"]
            self.assertEqual([r["seed"] for r in runs], [1, 2, 3, 4, 5])
            for run, h, expected in zip(
                runs, history["datasets"][dataset], results["datasets"][dataset]["runs"]
            ):
                self.assertEqual(len(h["epochs"]), 20)
                self.assertEqual(run["selected_epoch"], expected["selected_epoch"])
                self.assertEqual(
                    run["test_prediction_sha256"], expected["cache_sha256"]
                )
                self.assertEqual(
                    run["checkpoint_sha256"],
                    expected["checkpoint_sha256_from_completed_training"],
                )
                self.assertFalse(run["test_loaded_during_training"])
                total += len(h["epochs"])
        self.assertEqual(total, 300)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires model extras")
    def test_snapshot_excludes_unlisted_private_and_output_files(self):
        import tarc.runtime as runtime

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tarc").mkdir()
            (root / "private").mkdir()
            (root / "outputs").mkdir()
            (root / "tarc/model.py").write_text("pass\n")
            (root / "private/credentials.json").write_text('{"dummy": true}')
            (root / "outputs/history.json").write_text("{}")
            (root / "MANIFEST.sha256").write_text("0" * 64 + "  tarc/model.py\n")
            with patch.object(runtime, "__file__", str(root / "tarc/runtime.py")):
                runtime.source_snapshot(root / "snapshot")
            with zipfile.ZipFile(root / "snapshot/source_snapshot.zip") as archive:
                self.assertEqual(archive.namelist(), ["tarc/model.py"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires model extras")
    def test_encoder_lengths_reject_fractional_nonfinite_and_truncating_values(self):
        import numpy as np
        from tarc.data import FeatureDataset

        arrays = dict(
            ids=np.array(["example"]),
            y=np.array([0.0], dtype=np.float32),
            text=np.ones((1, 2, 3), dtype=np.float32),
            text_lengths=np.array([2]),
            text_input_ids=np.array([[1, 2, 3, 4]]),
            text_attention_mask=np.ones((1, 4), dtype=np.int64),
            text_token_type_ids=np.zeros((1, 4), dtype=np.int64),
            text_word_index=np.array([[-1, 0, 1, -1]]),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.npz"
            np.savez(path, **arrays, bert_lengths=np.array([4]))
            self.assertEqual(len(FeatureDataset(path, ["text"], "sentiment")), 1)
            for length in (2.5, float("nan"), 3):
                np.savez(path, **arrays, bert_lengths=np.array([length]))
                with self.assertRaises(ValueError):
                    FeatureDataset(path, ["text"], "sentiment")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires model extras")
    def test_encoder_conversion_round_trip_and_bad_digest(self):
        import torch
        from safetensors.torch import save_file
        from transformers import DebertaV2Config, AutoModel

        prepare = helper("prepare_encoder").prepare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = DebertaV2Config(
                vocab_size=16,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=12,
                max_position_embeddings=16,
            )
            model = AutoModel.from_config(config)
            config.save_pretrained(root)
            source = root / "initialization.safetensors"
            save_file(
                {"bert." + k: v.contiguous() for k, v in model.state_dict().items()},
                str(source),
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            prepare(source, root / "config.json", root / "converted", digest)
            restored = AutoModel.from_pretrained(
                root / "converted", local_files_only=True, use_safetensors=True
            )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, restored.state_dict()[key], rtol=0, atol=0
                )
            with self.assertRaises(ValueError):
                prepare(source, root / "config.json", root / "rejected", "0" * 64)


if __name__ == "__main__":
    unittest.main()
