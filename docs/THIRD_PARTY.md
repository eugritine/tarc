# Third-party components

The MIT License in the repository applies to TARC project source code. Dependencies,
backbone weights and datasets retain their own licenses and access terms.

| Component | Recorded version | Primary source |
|:--|:--|:--|
| PyTorch | 2.7.1 | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Transformers | 4.57.6 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| NumPy | 2.5.4 | https://pypi.org/pypi/numpy/2.5.4/json |
| SciPy | 1.18.1 | https://pypi.org/pypi/scipy/1.18.1/json |
| scikit-learn | 1.9.1 | https://pypi.org/pypi/scikit-learn/1.9.1/json |
| PyYAML | 6.0.3 | https://pypi.org/pypi/PyYAML/6.0.3/json |
| safetensors | 0.8.0 | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| DeBERTa-v3-large | Fixed revision in configs | https://huggingface.co/microsoft/deberta-v3-large |

This source repository does not bundle those dependency distributions. Inspect
the licenses and notices of the actual wheels/model files when redistributing
them; platform wheels may include separately licensed libraries. This inventory
covers the direct dependencies, not a complete transitive software bill of materials.

Dataset and feature-release terms are separate from the CMU SDK or upstream
feature-loader code license. No code license in this repository grants permission
to redistribute IEMOCAP or other restricted source/derived records.
