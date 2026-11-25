"""Verify the complete source release against MANIFEST.sha256."""

import hashlib
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    checked = 0
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
        ):
            raise ValueError(f"Unsafe manifest entry: {relative}")
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


if __name__ == "__main__":
    print(f"Verified {verify(Path(__file__).resolve().parents[1])} release files.")
