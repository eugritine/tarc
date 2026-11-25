# Environments

`requirements-cpu.lock.txt` records the installed dependency versions from the
isolated Linux x86_64 / Python 3.12.3 / PyTorch 2.7.1+cpu validation environment.
The original unit tests and synthetic task interfaces were executed there.
Dependency resolution was checked with pip in dry-run mode; a fresh installation
of this complete pin file has not been performed. This is a version-pinned record;
wheel hashes and all other operating-system
variants are not locked. It does not document a fresh five-seed training run.

```bash
python -m pip install -r environment/requirements-cpu.lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

The model experiments used PyTorch 2.7.1+cu128. CPU and CUDA builds are distinct;
do not install the CPU lock when intending to reproduce the CUDA environment.
Use the original model requirements and the official platform-specific PyTorch
instructions: https://pytorch.org/get-started/previous-versions/ . The original
CUDA environment still needs a complete platform-specific lock and numerical
reproducibility qualification before claiming cross-machine reproduction.
