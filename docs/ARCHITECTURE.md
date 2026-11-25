# Architecture and code map

1. **Online text encoding.** A supplied DeBERTa-v3-large encoder produces subword
   states; `pool_wordpieces_to_words` aligns them to the visible-word grid. Acoustic
   and visual inputs are the recorded provided features, with dataset-specific widths.
2. **Multiscale latent encoding.** `MultiScaleTemporalEncoder` projects each stream,
   applies sparse temporal processing at scales 1/2/4 and produces latent statistics.
3. **Shared source assessment.** `ReliabilityHead` uses uncertainty/agreement and
   configured contextual descriptors to compute source weights. Availability is
   explicit; an all-zero feature vector is not interpreted as a missing modality.
4. **Evidence construction.** `IntermodalCoupling` performs reliability-conditioned
   sparse interaction; `SparseExperts` constructs a pooled evidence descriptor.
5. **Controlled correction.** `readouts/unified.py` uses the primary text latent as
   anchor. IEMOCAP uses categorical correction; MOSI/MOSEI use polarity/magnitude
   correction. Source weights rho and correction gate q are different quantities.
6. **Training.** Endpoint losses supervise the text/evidence branches. Utility labels
   are detached error-improvement targets; gate input features are detached. Missing-
   modality consistency and the recorded core objective are retained.

## Readout equations

Categorical, when text is visible:

`p = (1 - q) p_text + q p_evidence`.

Scalar, when text is visible:

`r = (abs(t) + 0.5) * tanh(f)`

`s = t + q_sign * r`

`a = (1 - q_magnitude) * a_text + q_magnitude * a_evidence`

`prediction = tanh(s) * a`.

The diagnostic evidence endpoint is `tanh(f) * a_evidence`. Setting both scalar
correction gates to one is not the same endpoint. If auxiliary evidence is absent,
the correction gates are zero; if text is absent, the evidence branch is used.
Labels never enter the inference forward path.

The three tasks have independently fitted weights. A common evidence topology
does not mean that their checkpoints are interchangeable.
