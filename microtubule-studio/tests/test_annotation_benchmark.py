"""Check isolation and reproducibility of the annotation omission experiment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml


MODULE = Path(__file__).resolve().parents[1] / "training" / "prepare_annotation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("annotation_benchmark", MODULE)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class AnnotationBenchmarkTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        for split, count in (("train", 2), ("val", 1)):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            for index in range(count):
                image = np.zeros((32, 32), dtype=np.uint8)
                lines = []
                for (x, y), value in zip(((8, 8), (24, 8), (8, 24), (24, 24)), (10, 30, 60, 120)):
                    image[y - 3:y + 3, x - 3:x + 3] = value + index
                    lines.append(f"0 {x / 32} {y / 32} 0.1875 0.1875")
                lines += ["1 0.5 0.5 0.125 0.125", "2 0.5 0.5 1 1"]
                Image.fromarray(image).save(source / "images" / split / f"image{index}.png")
                # Preserve CRLF bytes in validation as part of the isolation check.
                raw = ("\r\n".join(lines) + "\r\n").encode()
                (source / "labels" / split / f"image{index}.txt").write_bytes(raw)
        data = source / "data.yaml"
        data.write_text(yaml.safe_dump(dict(path=str(source), train="images/train",
                                          val="images/val", names=BENCHMARK.NAMES)))
        return source, data

    def test_source_validation_and_other_classes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = self.fixture(root)
            snapshots = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            output = root / "experiment"
            result = BENCHMARK.prepare(data, output)
            self.assertEqual(result["omitted_training_cytosolic"], 2)
            self.assertEqual(result["after"]["train"][0], 6)
            for path, raw in snapshots.items():
                self.assertEqual(path.read_bytes(), raw)
            for split in ("train", "val"):
                for image in (source / "images" / split).glob("*.png"):
                    self.assertEqual(image.read_bytes(),
                                     (output / "dataset" / "images" / split / image.name).read_bytes())
                for label in (source / "labels" / split).glob("*.txt"):
                    new = output / "dataset" / "labels" / split / label.name
                    if split == "val":
                        self.assertEqual(label.read_bytes(), new.read_bytes())
                    else:
                        other = lambda p: [s for s in p.read_text().splitlines() if not s.startswith("0 ")]
                        self.assertEqual(other(label), other(new))

    def test_same_seed_reproduces_omissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, data = self.fixture(root)
            for name in ("a", "b"):
                BENCHMARK.prepare(data, root / name, seed=17)
            self.assertEqual((root / "a/training_annotation_scores.csv").read_bytes(),
                             (root / "b/training_annotation_scores.csv").read_bytes())

    def test_contrast_uses_surrounding_ring_and_handles_no_ring(self):
        image = np.full((16, 16), 10, dtype=np.float32)
        image[6:10, 6:10] = 30
        self.assertEqual(BENCHMARK.local_display_contrast(image, [0, 0.5, 0.5, 0.25, 0.25]), 20)
        self.assertIsNone(BENCHMARK.local_display_contrast(image, [0, 0.5, 0.5, 1, 1]))

    def test_invalid_input_fails_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = self.fixture(root)
            (source / "labels/train/image0.txt").write_text("0 0.5 0.5 nan 0.2\n")
            with self.assertRaises(ValueError):
                BENCHMARK.prepare(data, root / "experiment")
            self.assertFalse((root / "experiment").exists())


if __name__ == "__main__":
    unittest.main()
