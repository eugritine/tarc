import hashlib
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_checkpoint(path, state):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def source_snapshot(output):
    """Snapshot declared source files without traversing private/output directories."""
    root = Path(__file__).resolve().parents[1]
    manifest = root / "MANIFEST.sha256"
    if manifest.is_file():
        names = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
    else:
        # Installed wheels carry a distribution file list rather than the source manifest.
        from importlib.metadata import distribution

        names = [
            str(entry)
            for entry in distribution("tarc").files or []
            if str(entry).startswith("tarc/")
        ]
    folders = {"tarc", "configs", "scripts", "tests", "docs", ".github"}
    root_files = {
        "README.md",
        "DATA_FORMAT.md",
        "NOTICE.md",
        "LICENSE",
        "pyproject.toml",
        "requirements-model.txt",
        "start_demo.py",
        ".gitignore",
    }
    extensions = {".py", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".sh"}
    included = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source snapshot path")
        if not (
            name in root_files
            or (
                relative.parts
                and relative.parts[0] in folders
                and relative.suffix in extensions
            )
        ):
            continue
        path = root / relative
        if any(
            (root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        ):
            raise ValueError("Source snapshots cannot follow symbolic links")
        if not path.is_file():
            raise FileNotFoundError(name)
        included.append(path)
    if not included:
        raise ValueError("No declared source files available for the snapshot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "source_snapshot.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(included):
            archive.write(source, source.relative_to(root))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(seed, sample_id):
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def choose_device(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device
