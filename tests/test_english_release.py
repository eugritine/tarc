"""Check that public text files, filenames and decoded JSON contain no Han text."""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


class EnglishReleaseTests(unittest.TestCase):
    def test_public_text_and_filenames(self):
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if any(
                part in {".git", "__pycache__", ".venv", "build", "dist"}
                for part in relative.parts
            ):
                continue
            self.assertIsNone(HAN.search(relative.as_posix()), str(relative))
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            self.assertIsNone(HAN.search(text), str(relative))
            if path.suffix == ".json":
                for value in strings(json.loads(text)):
                    self.assertIsNone(HAN.search(value), str(relative))

    def test_viewer_language_and_class_order(self):
        self.assertIn('lang="en"', (ROOT / "demo/index.html").read_text())
        data = json.loads((ROOT / "demo/data.json").read_text())
        classes = data["datasets"]["iemocap"]
        self.assertEqual(
            classes["class_names"], ["happy_or_excited", "sad", "neutral", "angry"]
        )
        self.assertEqual(
            classes["class_display_names"],
            ["Happy / Excited", "Sad", "Neutral", "Angry"],
        )


if __name__ == "__main__":
    unittest.main()
