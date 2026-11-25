# Experiment assets

## Dataset identity

| Dataset | Train | Validation | Test | Source feature fingerprint |
|:--|--:|--:|--:|:--|
| MOSI | 1284 | 229 | 686 | See experiment_manifest.json |
| MOSEI | 16326 | 1871 | 4659 | See experiment_manifest.json |
| IEMOCAP | 3311 | 1031 | 1242 | See experiment_manifest.json |

Input-file hashes, tokenizer revision, source hashes and selected checkpoint/cache
hashes are recorded in [experiment_manifest.json](../assets/experiment_manifest.json).
These are fingerprints, not downloadable assets. IEMOCAP has 5,584 eligible samples
in this author-feature protocol. It is not the 5,531-sample raw-audio recipe and
uses a fixed session split rather than five-fold leave-one-session-out evaluation.

MOSI/MOSEI preserve the source feature split. Original BERT-visible word spans are
retokenized with the official slow DeBERTa SentencePiece tokenizer; the visible
word window and provided acoustic/visual features are unchanged. IEMOCAP uses
100-dimensional acoustic and 512-dimensional visual utterance features from the
DialogueCRN author-feature release. Labels 0 and 4 are merged into happy/excited;
1, 2 and 3 map to sad, neutral and angry; label 5 is excluded.

The raw-to-processed pipeline and the upstream feature-fitting scope are not fully
specified by the released code. Matching channel dimensions alone is insufficient
for reproducing these inputs. Do not substitute another preprocessing pipeline and
claim that it reproduces the recorded protocol.

## Shared initialization and encoder setup

The shared SST-2 initialization is a complete encoder state with `bert.*` keys.
Its identity is in the manifest. A generic DeBERTa download is not this initialization.
The upstream fixed revision may use a `.bin` weight file, whereas the training
entry expects a local safetensors encoder directory.

For an already obtained, trusted complete initialization, this helper produces a
local encoder directory without training:

```bash
python scripts/prepare_encoder.py \
  --initialization pretrained/sst2_initialization.safetensors \
  --config configs/encoder_config.json --output pretrained/deberta-v3-large \
  --expected-sha256 143d2436a0402dfcc4960bcbacc53d83699c2f267b224c944e58c2a1ed6d7365
```

Use `--validate-only` to check the digest and encoder key/shape coverage without
writing output weights. The generated encoder contains SST-2-initialized weights;
it is not a byte-identical copy of the original generic backbone file. Training
strictly loads the same shared initialization afterward. This helper does not
recreate SST-2 pretraining or provide a download URL for its result.

## Recompute the archived main table

With the original prediction caches available locally:

```bash
python scripts/recompute_test_table.py \
  --sentiment-dir /path/to/sentiment/test_evaluation \
  --emotion-dir /path/to/emotion/test_evaluation/iemocap \
  --output outputs/recomputed_main_table.json
```

The sentiment folder must contain `mosi_seed_1.npz` through `mosi_seed_5.npz` and
`mosei_seed_1.npz` through `mosei_seed_5.npz`; the emotion folder must contain
`seed_1.npz` through `seed_5.npz`. SHA-256 is checked against the registered seed
before reading and scoring each cache. This recomputes stored predictions, not
fresh model inference. The caches and their target labels are not distributed
with this supplement.

Original per-epoch validation metrics and selected epochs are provided in
[training_histories.json](../results/training_histories.json). History and completion
receipt digests bind those records to their original experiment artifacts.

## Evaluate newly completed runs

After training seeds 1–5 and selecting their validation checkpoints:

```bash
bash scripts/evaluate_five_seeds.sh mosi data/mosi outputs/mosi cuda
```

This runs test prediction, evaluation and aggregation for all five seeds. It
refuses to overwrite existing prediction/metric files and does not run training.
The general evaluation CLI accepts caller-supplied seed/split metadata; it does
not independently certify a run's identity. For the archived main table, use the
fingerprint-bound cache checker above.

## Asset access

The repository does not currently provide public download URLs for the processed
splits, shared initialization or final checkpoints. The project code license does
not grant access or redistribution rights for dataset/feature/model assets.
Acquire those assets under their original terms. Dataset source identifiers and
fingerprints do not substitute for permission to redistribute per-example records.
