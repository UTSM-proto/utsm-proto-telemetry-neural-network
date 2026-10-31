import ast
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_clean_notebook_and_exact_bundled_sources(self):
        nb = json.loads((ROOT/"notebooks/retrain_laval.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len({c["id"] for c in nb["cells"]}), len(nb["cells"]))
        assignments = {}
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = node.value.value
        self.assertEqual(assignments["BUNDLED_ENVIRONMENT_SOURCE"], (ROOT/"colab_environment.py").read_text(encoding="utf-8"))
        source = (ROOT/"laval_training.py").read_text(encoding="utf-8")
        self.assertEqual(assignments["BUNDLED_TRAINING_SOURCE"], source)
        self.assertEqual(assignments["BUNDLED_SOURCE_SHA256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(assignments["REQUIREMENTS"], (ROOT/"requirements-colab.txt").read_text())
        self.assertEqual(assignments["FRONT_CAMPUS_REFERENCE_JSON"],
                         (ROOT/"configs/front_campus_reference.json").read_text(encoding="utf-8"))
        self.assertNotIn("raw.githubusercontent.com", "".join("".join(c["source"]) for c in nb["cells"]))
        self.assertEqual(nb["metadata"]["utsm_training_source_sha256"], assignments["BUNDLED_SOURCE_SHA256"])


if __name__ == "__main__":
    unittest.main()
