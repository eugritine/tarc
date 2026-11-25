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
    root = Path(__file__).resolve().parents[1]
    path = Path(output) / "source_snapshot.zip"
    included = []
    excluded = {
        "runs",
        "data",
        "__pycache__",
        ".pytest_cache",
        "verification",
        "validation",
        ".venv",
        "venv",
        "env",
        ".git",
        "historical",
        "backups",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
    }
    for directory, dirs, names in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in excluded
            and not name.endswith(".egg-info")
            and not (Path(directory) / name).is_symlink()
        ]
        for name in names:
            p = Path(directory) / name
            if (
                p.is_file()
                and not p.is_symlink()
                and p.suffix
                in {".py", ".yaml", ".toml", ".txt", ".json", ".md", ".sh", ".ps1"}
            ):
                included.append(p)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(included):
            z.write(p, p.relative_to(root))
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
