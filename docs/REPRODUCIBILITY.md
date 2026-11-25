# Reproducibility

## Inputs

MOSI and MOSEI use word-aligned text and acoustic/visual sequences. IEMOCAP uses
utterance-level acoustic/visual features. All tasks use online DeBERTa-v3-large
text encoding. See [DATA_FORMAT.md](../DATA_FORMAT.md) for arrays and alignment.

| Dataset | Online text width | Audio width | Visual width | Output |
|:--|--:|--:|--:|:--|
| MOSI | 1024 | 5 | 20 | sentiment score |
| MOSEI | 1024 | 74 | 35 | sentiment score |
| IEMOCAP | 1024 | 100 | 512 | four-class probabilities |

The cached text carrier width is dataset-specific and is not the online encoder
width. Normalization statistics are fitted on training data only.

## Training protocol

| Setting | Value |
|:--|:--|
| Seeds | 1, 2, 3, 4, 5 |
| Epochs | 20 |
| Batch size | 16 |
| Precision | FP32 |
| Optimizer | AdamW |
| Encoder learning rate | 1e-5 |
| Other parameters learning rate | 1e-4 |
| Weight decay | 0.01 |
| Linear warmup | 10% of total updates |
| Gradient clipping | 1.0 |
| Trainable encoder layers | 24 |
| Modality dropout | 0.3, with nonempty input masks |
| Endpoint / utility loss weights | 0.25 / 0.1 |
| Utility temperature | 0.25 |
| Initial correction coefficient | 0.1 |

MOSI/MOSEI use video-balanced replacement sampling; IEMOCAP uses shuffled sampling.
Complete model and loss configurations are in `configs/`.

Sentiment checkpoint selection prefers validation MAE <= 0.82, then higher Acc-2,
then lower MAE. IEMOCAP selects higher validation weighted F1, with accuracy as a
tie-breaker. Every epoch is run; the training runner does not evaluate test data.
The configuration field `patience` does not activate early stopping.

Training outputs include the best checkpoint and validation predictions, epoch
metrics, sampled index streams, train-only normalization statistics, configuration,
environment metadata and a source snapshot.

## Evaluation protocol

MOSI/MOSEI Acc-2 and weighted F1 place label zero in the nonnegative class. Accuracy
and F1 are percentages; MAE is in the original label units. Five seeds are summarized
using sample standard deviation (`ddof=1`).

IEMOCAP class order is `happy_or_excited, sad, neutral, angry`, using sessions 1–3
for training, 4 for validation and 5 for testing. Its W-Acc is sample accuracy.
This is a fixed-session evaluation, not a five-fold leave-one-session-out protocol.

`python -m tarc.evaluate` accepts labeled prediction JSON or NPZ (`ids`, `y`, `logits`).
Genuine labels are required for metrics. A placeholder `y` used for unlabeled
inference must not be evaluated as a dataset annotation.

## Required assets

Training requires the original processed datasets, DeBERTa encoder and shared
SST-2 initialization. Inference requires a complete trained checkpoint and matching
processed features. Those assets are not included, and no public download links
are currently provided for the datasets, shared initialization or final checkpoints.
The generic DeBERTa backbone does not replace the shared initialization.

Reported per-seed values are provided in `results/final_results.json`. Running
`scripts/summarize_archived_results.py` checks their means and sample standard
deviations; this only recomputes summary statistics, not model predictions.
