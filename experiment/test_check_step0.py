"""检查回归工具能否识别损坏的数据和数值变化。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import check_step0 as check


class CheckStep0Test(unittest.TestCase):
    def test_arrays(self):
        a = np.array([0.0, 1.0])
        self.assertTrue(check.compare_arrays(a, a.copy(), 0)["ok"])
        self.assertFalse(check.compare_arrays(a, np.array([-0.0, 1.0]), 0)["ok"])
        self.assertFalse(check.compare_arrays(a, a.astype(np.float32), 0)["ok"])
        self.assertFalse(check.compare_arrays(a, a[:1], 0)["ok"])
        for bad in (np.nan, np.inf, -np.inf):
            self.assertFalse(check.compare_arrays(a, np.array([bad, 1.0]), 0)["ok"])
        self.assertTrue(check.compare_arrays(a, a + 1e-6, 1e-5)["ok"])
        self.assertFalse(check.compare_arrays(a, a + 1e-4, 1e-5)["ok"])

    def test_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base", root / "candidate"
            base.mkdir()
            candidate.mkdir()
            meta = np.frombuffer(json.dumps({"t": 0.1, "step": 1, "dt": 0.1}).encode(), dtype=np.uint8)
            np.savez(base / "a.npz", cons_0=np.array([1.0]), _meta=meta)
            digest = hashlib.sha256((base / "a.npz").read_bytes()).hexdigest()
            (base / "case.json").write_text(json.dumps({"files": [{"name": "a.npz", "sha256": digest}]}))
            def compare(**fields):
                np.savez(candidate / "a.npz", **fields)
                return check.compare_case(base, candidate, 0)["status"]
            self.assertEqual(compare(cons_0=np.array([1.0]), _meta=meta), "PASS")
            self.assertEqual(compare(_meta=meta), "FAIL")
            self.assertEqual(compare(cons_0=np.array([1.0]), extra=np.array([1]), _meta=meta), "FAIL")
            self.assertEqual(compare(cons_0=np.array([1.0])), "FAIL")
            self.assertEqual(compare(cons_0=np.array([1.0]), _meta=np.array([1], dtype=np.uint8)), "FAIL")
            missing = np.frombuffer(b'{"t": 0.1, "step": 1}', dtype=np.uint8)
            self.assertEqual(compare(cons_0=np.array([1.0]), _meta=missing), "FAIL")
            np.savez(base / "a.npz", cons_0=np.array([2.0]), _meta=meta)
            self.assertEqual(compare(cons_0=np.array([2.0]), _meta=meta), "FAIL")


if __name__ == "__main__":
    unittest.main()
