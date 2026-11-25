# TARC

## Text-Anchored Reliability-Aware Correction for Multimodal Sentiment Analysis

PyTorch implementation of TARC for multimodal sentiment analysis and emotion
recognition on **MOSI**, **MOSEI**, and **IEMOCAP**.

TARC uses a text prediction as an anchor and learns how much multimodal evidence
should change it. Multiscale latent encoding, shared source assessment and sparse
cross-modal interaction construct the evidence; utility-supervised correction
controls its contribution to the output. See [Architecture](docs/ARCHITECTURE.md)
for the modules and task-specific readouts.

## Environment

- Python 3.12
- PyTorch 2.7.1
- Transformers 4.57.6

Install from the repository root:

```bash
python -m pip install -e ".[model]"
```

The experiments used PyTorch 2.7.1+cu128. Select a PyTorch installation compatible
with your device. Full dependency versions are in [requirements-model.txt](requirements-model.txt).

## Data preparation

Prepare tokenized text and acoustic/visual features according to
[DATA_FORMAT.md](DATA_FORMAT.md):

```text
data/
  mosi/       train.npz, val.npz, test.npz
  mosei/      train.npz, val.npz, test.npz
  iemocap/    train.npz, val.npz, test.npz
pretrained/
  deberta-v3-large/       encoder weights and config
  sst2_initialization.safetensors
```

The text backbone is [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large).
Training also uses a shared SST-2 encoder initialization. The dataset files,
shared initialization and fine-tuned checkpoints are not included, and download
links for those assets are not currently available. The original feature and token
alignment specifications are required; generic dataset downloads are not a direct
replacement. Configuration and split details are in [Reproducibility](docs/REPRODUCIBILITY.md).

## Training

```bash
python -m tarc.train \
  --dataset mosi --data data/mosi \
  --encoder pretrained/deberta-v3-large \
  --initialization pretrained/sst2_initialization.safetensors \
  --output outputs/mosi/seed_1 --seed 1 --device cuda
```

Use `--dataset mosei` or `--dataset iemocap` with the corresponding data directory
and output directory. Configurations are in `configs/`. Each run trains for 20 epochs
and selects its checkpoint on validation. To run seeds 1–5:

```bash
bash scripts/train_five_seeds.sh mosi data/mosi \
  pretrained/deberta-v3-large pretrained/sst2_initialization.safetensors \
  outputs/mosi cuda
```

## Evaluation

Run a trained checkpoint on a labeled processed split, then calculate metrics:

```bash
python -m tarc.predict --dataset mosi \
  --checkpoint outputs/mosi/seed_1/best.pt --features data/mosi/test.npz \
  --device cuda --output outputs/mosi/seed_1/test_predictions.json

python -m tarc.evaluate --dataset mosi --seed 1 --split test \
  --predictions outputs/mosi/seed_1/test_predictions.json \
  --output outputs/mosi/seed_1/test_metrics.json

python -m tarc.aggregate --inputs outputs/mosi/seed_*/test_metrics.json \
  --output outputs/mosi/summary.json
```

## Results

Mean ± sample standard deviation over five seeds:

| Dataset | Acc-2 / W-Acc (%) | W-F1 (%) | MAE |
|:--|--:|--:|--:|
| MOSI | 88.16 ± 0.48 | 88.14 ± 0.50 | 0.555 ± 0.021 |
| MOSEI | 86.56 ± 0.31 | 86.53 ± 0.34 | 0.505 ± 0.009 |
| IEMOCAP | 73.67 ± 1.14 | 73.86 ± 1.08 | — |

MOSI/MOSEI Acc-2 and weighted F1 use the including-zero convention. IEMOCAP W-Acc
is sample accuracy on four classes, using sessions 1–3 / 4 / 5 for train/validation/test.
Per-seed results are in [results/final_results.json](results/final_results.json).

## Prediction visualization

An optional [prediction viewer](demo/README.md) compares text, evidence and
corrected outputs for the saved validation samples. It can be run locally or
published with the included GitHub Pages workflow.

## Acknowledgements

The language encoder uses [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa)
through [Transformers](https://github.com/huggingface/transformers).
