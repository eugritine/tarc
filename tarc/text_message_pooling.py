"""Optional text-query message representation with the original no-text fallback.

Text still conditions queries, reliability and expert routing. This is neither
an independent AV-only branch nor a causal-effect estimator. The shared h/z
change affects fused sign, magnitude and consistency, not just one loss head.
"""

import torch
from .ops import masked_mean


def text_message_pool(coupled_text, original_text, mask, text_visible, legacy_h):
    message = masked_mean(coupled_text - original_text, mask)
    return torch.where(text_visible[:, None], message, legacy_h)
