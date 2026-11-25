"""Task readouts for the selected TARC core; no alternative fusion backbone.

Categorical execution is inherited unchanged. Sentiment keeps the same primary
text latent and post-expert evidence descriptor, with scalar polarity/magnitude
heads and the already implemented dual-utility correction equations.
"""

import torch
from torch import nn
from tarc.ops import masked_mean
from tarc.readouts.selective import SelectiveTARC
from tarc.readouts.correction import SelectiveSentimentCorrection


class UnifiedReadoutTARC(SelectiveTARC):
    def __init__(self, core):
        if core.cfg.task == "classification":
            super().__init__(core)
            return
        if core.cfg.task != "sentiment":
            raise ValueError(
                "Only the selected categorical core and its sentiment readout are supported"
            )
        nn.Module.__init__(self)
        self.core = core
        self.cfg = core.cfg
        self.text_index = self.cfg.modalities.index("text")
        self.sentiment = True
        self.text_sign_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.magnitude_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.correction = SelectiveSentimentCorrection(
            self.cfg.hidden_dim, self.cfg.hidden_dim
        )

    def decode(self, encoded, available=None, intervention="R_full"):
        if not self.sentiment:
            return super().decode(encoded, available, intervention)
        base = self.core.decode(encoded, available, intervention)
        text = masked_mean(encoded["primary"]["z"]["text"], encoded["masks"]["text"])
        visible = base["available"][:, self.text_index : self.text_index + 1]
        other = [i for i in range(len(self.cfg.modalities)) if i != self.text_index]
        evidence = base["available"][:, other].any(-1, keepdim=True)
        out = self.correction(
            text,
            base["z"],
            self.text_sign_head(text),
            base["logits"],
            self.magnitude_head(base["z"]),
            visible,
            evidence,
        )
        return {**base, **out}
