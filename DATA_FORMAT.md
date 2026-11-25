# Final TARC processed NPZ input

Use the same preprocessing and feature releases as the final experiments.
These files contain numeric arrays only (`allow_pickle=False`). `y` is used for
training/evaluation or output comparison; it does not enter the forward model.
For unlabeled inference, a compatible placeholder `y` array is required by the
shared dataset loader and must not be presented as a ground-truth annotation.

Required general arrays:

| Key | Shape/type | Meaning |
|---|---|---|
| `ids` | `[N]` Unicode/bytes | Unique sample IDs |
| `y` | `[N]` numeric | Sentiment float or four-class integer |
| `text` | `[N,T,D_carrier]` float32 | Original cached carrier on the visible-word/utterance grid; overwritten by online encoder states |
| `audio` | `[N,T_A,D_A]` float32 | Provided acoustic features |
| `vision` | `[N,T_V,D_V]` float32 | Provided visual features |
| `text_lengths`, `audio_lengths`, `vision_lengths` | `[N]` integer | Valid prefix lengths, explicitly stored |
| `available` | `[N,3]` bool, optional | T/A/V availability; nonempty per observation |

Required final online-language arrays:

| Key | Shape/type | Meaning |
|---|---|---|
| `text_input_ids` | `[N,S]` integer | DeBERTa token IDs |
| `text_attention_mask` | `[N,S]` binary | Valid subword positions |
| `text_token_type_ids` | `[N,S]` integer | Token types |
| `text_word_index` | `[N,S]` signed integer | Zero-based visible-word index; `-1` for special/padded tokens |
| `bert_lengths` | `[N]` integer | Valid encoder grid lengths |

Optional `{modality}_mask` and `{modality}_time` arrays specify observed positions
and sorted normalized timestamps in `[0,1]`. Absence masks cannot be inferred
from all-zero feature values. Use the checkpoint's training-only normalizer at
inference; do not refit normalization on an incoming validation/test split.

IEMOCAP has one genuine acoustic/visual utterance position (100/512 channels).
MOSI/MOSEI acoustic/visual dimensions are 5/20 and 74/35. No raw-stream feature
extractor is substituted by these wrappers. The complete loader validation is in
`tarc/data.py`.

`D_carrier` must follow the original processed file and its checkpoint training
normalizer; it is not universally 1024 (IEMOCAP carriers have width 100). The
online DeBERTa encoder output supplied to the core has width 1024.
