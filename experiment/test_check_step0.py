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

    def test_step_filename_mapping(self):
        """旧 _step 基准与新 -istep- 候选按 step 一对一配对，不改基准文件名。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base", root / "candidate"
            base.mkdir()
            candidate.mkdir()
            meta = np.frombuffer(
                json.dumps({"t": 0.1, "step": 1, "dt": 0.1}).encode(),
                dtype=np.uint8)
            np.savez(base / "channel_step000001.npz", cons_0=np.array([1.0]),
                     _meta=meta)
            digest = hashlib.sha256(
                (base / "channel_step000001.npz").read_bytes()).hexdigest()
            (base / "case.json").write_text(json.dumps(
                {"files": [{"name": "channel_step000001.npz",
                            "sha256": digest}]}))
            np.savez(candidate / "OCFD3d-Mesh-istep-00000001.npz",
                     cons_0=np.array([1.0]), _meta=meta)
            self.assertEqual(check.compare_case(base, candidate, 0)["status"], "PASS")
            # 同一步号多一个候选文件 → 无法一对一
            np.savez(candidate / "other-istep-00000001.npz",
                     cons_0=np.array([1.0]), _meta=meta)
            self.assertEqual(check.compare_case(base, candidate, 0)["status"], "FAIL")
            (candidate / "other-istep-00000001.npz").unlink()
            # 候选多出无法映射的步号
            np.savez(candidate / "extra-istep-00000002.npz",
                     cons_0=np.array([1.0]), _meta=meta)
            self.assertEqual(check.compare_case(base, candidate, 0)["status"], "FAIL")
            (candidate / "extra-istep-00000002.npz").unlink()
            # 文件名步号与 _meta.step 不一致
            bad = np.frombuffer(
                json.dumps({"t": 0.1, "step": 2, "dt": 0.1}).encode(),
                dtype=np.uint8)
            np.savez(candidate / "OCFD3d-Mesh-istep-00000001.npz",
                     cons_0=np.array([1.0]), _meta=bad)
            result = check.compare_case(base, candidate, 0)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("_meta.step" in p for p in result["problems"]))

    def test_old_baseline_against_new_candidate(self):
        """基准是第八步以前的格式（cons_0..4）、候选是第八步起的格式（b0）：基准的
        守恒量先换成原始量再逐位比；值差一点判 FAIL。"""
        from opencfd_jax.src import io
        rng = np.random.default_rng(0)
        cons = rng.uniform(0.5, 1.5, (5, 2, 3, 4))
        meta = {"t": 0.1, "step": 1, "dt": 0.1, "gamma": 1.4, "Ama": 0.1}
        meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        prim = io.primitive_of(cons, io.phys_from_meta(meta))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base", root / "candidate"
            base.mkdir()
            candidate.mkdir()
            np.savez(base / "a.npz", **{f"cons_{i}": cons[i] for i in range(5)},
                     _meta=meta_bytes)
            digest = hashlib.sha256((base / "a.npz").read_bytes()).hexdigest()
            (base / "case.json").write_text(json.dumps(
                {"files": [{"name": "a.npz", "sha256": digest}]}))

            def compare(field):
                np.savez(candidate / "a.npz", b0=field, _meta=meta_bytes)
                return check.compare_case(base, candidate, 0)["status"]

            self.assertEqual(compare(prim), "PASS")
            damaged = prim.copy()
            damaged[4, 1, 1, 1] = np.nextafter(damaged[4, 1, 1, 1], np.inf)
            self.assertEqual(compare(damaged), "FAIL")

    def test_npz_summary_reads_both_layouts(self):
        """登记清单：新格式按 b0 记一项，密度取 b0 的第 0 个分量；旧的取 cons_0。"""
        import run_step0
        rng = np.random.default_rng(1)
        cons = rng.uniform(0.5, 1.5, (5, 2, 3, 4))
        meta_bytes = np.frombuffer(b'{"t": 0.1, "step": 1, "dt": 0.1}', dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old.npz", Path(tmp) / "new.npz"
            np.savez(old, **{f"cons_{i}": cons[i] for i in range(5)}, _meta=meta_bytes)
            np.savez(new, b0=cons, _meta=meta_bytes)        # 只看数组的排法，不求解
            for path, keys in ((old, ["cons_0", "cons_1", "cons_2", "cons_3", "cons_4"]),
                               (new, ["b0"])):
                summary = run_step0.npz_summary(path)
                self.assertEqual(sorted(summary["fields"]), keys)
                self.assertEqual(summary["rho"], [float(cons[0].min()), float(cons[0].max())])
            # 这一步以前登记的清单没有 rho 一项：报告照旧从 cons_0 的统计取密度范围
            manifest = {"created": "", "env": {"python": "", "numpy": ""}, "git": {},
                        "cases": [{"id": "x", "title": "t", "wall_seconds": 1.0, "files": [
                            {"bytes": 10, "fields": {"cons_0": {"min": 0.9, "max": 1.1}}},
                            {"bytes": 10, "rho": [0.95, 1.05]}]}]}
            report = Path(tmp) / "report.md"
            run_step0.write_markdown(manifest, report)
            self.assertIn("| 0.9 ~ 1.1 |", report.read_text(encoding="utf-8"))

    def test_step_of_supports_both_output_names(self):
        self.assertEqual(check.step_of("channel_step000001.npz"), 1)
        self.assertEqual(check.step_of("OCFD3d-Mesh-istep-00000001.npz"), 1)
        self.assertEqual(check.step_of("OCFD3d-Mesh-istep-000000001.npz"), 1)
        self.assertIsNone(check.step_of("channel.npz"))


if __name__ == "__main__":
    unittest.main()
