"""Online language-encoder interface and word-aligned pooling for TARC."""

from __future__ import annotations

import torch
from transformers import BertModel

from .model import TARC


def pool_wordpieces_to_words(hidden, word_index, word_slots):
    """Mean-pool BERT wordpieces onto an explicit zero-based word grid."""
    if hidden.shape[:2] != word_index.shape:
        raise ValueError("wordpiece map must match BERT sequence shape")
    if (
        word_slots < 1
        or torch.any(word_index < -1)
        or torch.any(word_index >= word_slots)
    ):
        raise ValueError("wordpiece map contains an invalid word slot")
    valid = word_index >= 0
    index = word_index.clamp_min(0)
    pooled = hidden.new_zeros(hidden.shape[0], word_slots, hidden.shape[-1])
    pooled.scatter_add_(
        1,
        index[:, :, None].expand_as(hidden),
        hidden * valid[:, :, None].to(hidden.dtype),
    )
    counts = hidden.new_zeros(hidden.shape[0], word_slots)
    counts.scatter_add_(1, index, valid.to(hidden.dtype))
    pooled = pooled / counts.clamp_min(1)[:, :, None]
    return pooled, counts > 0


class TARCTextEncoder(TARC):
    """Use last-layer BERT wordpiece states as TARC's text sequence.

    The final configuration supplies a DeBERTa-v3-large encoder explicitly.
    """

    def __init__(self, cfg, bert_path=None, train_last_n=None, bert_model=None):
        super().__init__(cfg)
        if "text" not in cfg.modalities:
            raise ValueError("TARC-BERT-FT requires the text modality")
        if bert_model is None:
            if not bert_path:
                raise ValueError("bert_path is required")
            bert_model = BertModel.from_pretrained(
                bert_path,
                local_files_only=True,
                use_safetensors=True,
                add_pooling_layer=False,
            )
        self.bert = bert_model
        if self.bert.config.hidden_size != cfg.input_dims["text"]:
            raise ValueError(
                "BERT hidden size must match the existing TARC text projection"
            )
        layers = len(self.bert.encoder.layer)
        if train_last_n is None:
            train_last_n = layers
        if not 1 <= train_last_n <= layers:
            raise ValueError("train_last_n must be within the BERT encoder depth")
        self.train_last_n = int(train_last_n)
        self._set_trainable_layers()

    def _set_trainable_layers(self):
        cutoff = len(self.bert.encoder.layer) - self.train_last_n
        for p in self.bert.embeddings.parameters():
            p.requires_grad = cutoff == 0
        for i, layer in enumerate(self.bert.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = i >= cutoff

    def _enforce_frozen_eval(self):
        cutoff = len(self.bert.encoder.layer) - self.train_last_n
        if cutoff > 0:
            self.bert.embeddings.eval()
            for layer in self.bert.encoder.layer[:cutoff]:
                layer.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._enforce_frozen_eval()
        return self

    def encode(self, batch):
        if "bert" not in batch:
            raise ValueError("TARC-BERT-FT requires tokenized BERT inputs")
        b = batch["bert"]
        output = self.bert(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
            token_type_ids=b["token_type_ids"],
            return_dict=True,
        )
        hidden = output.last_hidden_state
        if "word_index" in b:
            hidden, mapped = pool_wordpieces_to_words(
                hidden, b["word_index"], batch["mask"]["text"].shape[1]
            )
            if not torch.equal(mapped, batch["mask"]["text"]):
                raise ValueError("wordpiece map and downstream word mask disagree")
        if hidden.shape[:2] != batch["mask"]["text"].shape:
            raise ValueError("BERT token grid and downstream text mask differ")
        x = dict(batch["x"])
        x["text"] = hidden
        replaced = {**batch, "x": x}
        return super().encode(replaced)

    def parameter_counts(self):
        bert_total = sum(p.numel() for p in self.bert.parameters())
        bert_trainable = sum(
            p.numel() for p in self.bert.parameters() if p.requires_grad
        )
        tarc_total = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("bert.")
        )
        tarc_trainable = sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith("bert.") and p.requires_grad
        )
        return {
            "bert_total": bert_total,
            "bert_trainable": bert_trainable,
            "tarc_total": tarc_total,
            "tarc_trainable": tarc_trainable,
            "complete_total": bert_total + tarc_total,
            "complete_trainable": bert_trainable + tarc_trainable,
        }
