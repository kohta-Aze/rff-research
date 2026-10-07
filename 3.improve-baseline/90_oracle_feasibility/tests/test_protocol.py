"""Regression checks for provenance, source leakage and archive handling."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, inside, load_config, read_json, write_json
from download_data import acquire, extract
from prepare_data import inventory, partition_groups, prepare, roles, validate_plan
from prepare_within_record import prepare_within_record


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = deepcopy(load_config(ROOT / "configs/oracle_iq.json"))
        self.config["split"].update(expected_classes=4, known_classes=2,
                                    validation_unknown_classes=1, test_unknown_classes=1)
        self.config["input"]["max_windows_per_recording"] = 4

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, runs=6, samples=2048):
        raw = self.root / "raw"
        raw.mkdir()
        write_json(raw / "SYNTHETIC_FIXTURE.json", {"synthetic": True})
        for run in range(runs):
            for iq in range(4):
                rng = np.random.default_rng(run * 100 + iq)
                wave = rng.normal(size=samples) + 1j * rng.normal(size=samples)
                stem = raw / f"Demod_WiFi_cable_X310_TEST_IQ#{iq + 1}_run{run + 1}"
                wave.astype("<c16").tofile(stem.with_suffix(".sigmf-data"))
                write_json(stem.with_suffix(".sigmf-meta"), {"global": {"core:datatype": "cf32"}})
        return raw

    def test_complex128_override_and_class_label(self):
        records, errors = inventory(self.fixture())
        self.assertFalse(errors)
        self.assertEqual(len(records), 24)
        self.assertEqual(records[0]["effective_dtype"], "<c16")
        self.assertEqual(records[0]["sample_count"], 2048)
        self.assertEqual(records[0]["declared_dtype"], "cf32")

    def test_duplicate_raw_content_is_rejected(self):
        raw = self.fixture()
        paths = sorted(raw.glob("*.sigmf-data"))
        paths[1].write_bytes(paths[0].read_bytes())
        _, errors = inventory(raw)
        self.assertTrue(any("Duplicate raw content" in error["error"] for error in errors))

    def test_official_legacy_metadata_wrapper_is_supported(self):
        raw = self.fixture()
        for path in raw.glob("*.sigmf-meta"):
            write_json(path, {"_metadata": read_json(path), "data_file": "C:/untrusted/original/path"})
        records, errors = inventory(raw)
        self.assertFalse(errors)
        self.assertEqual(len(records), 24)
        self.assertTrue(all(row["metadata_layout"] == "legacy_wrapper" for row in records))

    def within_config(self):
        config = deepcopy(self.config)
        config["scope"] = "exploratory_within_record_injected_configuration_not_cross_day"
        config["split"].update(unit="time_block", guard_windows_each_side=2,
                               max_windows_per_block={"train": 8, "validation": 4, "test": 4})
        return config

    def test_within_record_blocks_are_explicit_and_nonoverlapping(self):
        raw = self.fixture(runs=1, samples=256 * 128)
        config = self.within_config()
        with self.assertRaisesRegex(ValueError, "strict preparer"):
            prepare(raw, self.root / "strict", config)
        result = prepare_within_record(raw, self.root / "within", config)
        self.assertTrue(result["verification_passed"])
        self.assertFalse(result["independent_recording_evaluation"])
        self.assertEqual(result["cross_partition_source_overlap"], 2)
        self.assertEqual(result["overlapping_windows"], 0)

    def test_within_record_wrong_time_block_is_rejected(self):
        raw = self.fixture(runs=1, samples=256 * 128)
        prepared = self.root / "within"
        prepare_within_record(raw, prepared, self.within_config())
        plan = read_json(prepared / "split.json")
        label = plan["views"]["test_known"][0]["iq_configuration"]
        plan["views"]["test_known"][0] = deepcopy(next(row for row in plan["views"]["train"]
                                                      if row["iq_configuration"] == label))
        with self.assertRaisesRegex(ValueError, "assigned time block"):
            validate_plan(prepared, plan)

    def test_within_record_cannot_claim_independent_evaluation(self):
        raw = self.fixture(runs=1, samples=256 * 128)
        prepared = self.root / "within"
        prepare_within_record(raw, prepared, self.within_config())
        plan = read_json(prepared / "split.json")
        plan["independent_recording_evaluation"] = True
        with self.assertRaisesRegex(ValueError, "exploratory scope"):
            validate_plan(prepared, plan)

    def test_within_record_short_recording_is_rejected(self):
        raw = self.fixture(runs=1)
        with self.assertRaisesRegex(ValueError, "too short"):
            prepare_within_record(raw, self.root / "within", self.within_config())

    def test_publisher_checksum_is_checked(self):
        raw = self.fixture()
        meta = next(raw.glob("*.sigmf-meta"))
        write_json(meta, {"global": {"core:datatype": "cf32", "core:sha512": "0" * 128}})
        _, errors = inventory(raw)
        self.assertTrue(any("sha512 mismatch" in error["error"] for error in errors))

    def test_no_window_fallback_for_too_few_groups(self):
        with self.assertRaisesRegex(ValueError, "At least 3"):
            partition_groups(self.config, [{"split_group": "a"}, {"split_group": "b"}])

    def test_source_groups_and_unknown_roles_are_disjoint(self):
        raw = self.fixture()
        result = prepare(raw, self.root / "prepared", self.config)
        self.assertTrue(result["verification_passed"])
        self.assertTrue(result["synthetic_fixture"])
        plan = read_json(self.root / "prepared/split.json")
        training = {r["split_group"] for r in plan["views"]["train"]}
        testing = {r["split_group"] for r in plan["views"]["test_known"]}
        self.assertFalse(training & testing)
        self.assertFalse(set(plan["class_roles"]["validation_unknown"]) & set(plan["class_roles"]["test_unknown"]))

    def test_group_leakage_tampering_is_rejected(self):
        raw = self.fixture()
        prepared = self.root / "prepared"
        prepare(raw, prepared, self.config)
        plan = read_json(prepared / "split.json")
        plan["views"]["test_known"][0]["split_group"] = plan["views"]["train"][0]["split_group"]
        with self.assertRaisesRegex(ValueError, "Source leakage"):
            validate_plan(prepared, plan)

    def test_prepared_modification_is_rejected(self):
        raw = self.fixture()
        prepared = self.root / "prepared"
        prepare(raw, prepared, self.config)
        plan = read_json(prepared / "split.json")
        path = prepared / plan["views"]["train"][0]["prepared_path"]
        with path.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_plan(prepared, plan)

    def test_zip_traversal_is_rejected(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../escape.sigmf-data", b"data")
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            extract(archive, self.root / "extracted", 1024)
        self.assertFalse((self.root / "escape.sigmf-data").exists())

    def test_zip_size_cap_is_enforced(self):
        archive = self.root / "large.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("sample.sigmf-data", b"x" * 1024)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            extract(archive, self.root / "extracted", 32)

    def test_html_cannot_be_saved_as_archive(self):
        response = io.BytesIO(b"<html>blocked</html>")
        response.headers = {"Content-Type": "text/html"}
        with patch("download_data.request", return_value=response):
            with self.assertRaisesRegex(ValueError, "Expected an archive"):
                acquire("https://example.test/data", self.root / "a.zip", 1024)
        self.assertFalse((self.root / "a.zip").exists())

    def test_existing_non_archive_is_rejected(self):
        archive = self.root / "a.zip"
        archive.write_text("<html>not data</html>")
        with self.assertRaisesRegex(ValueError, "not a readable ZIP/TAR"):
            acquire("https://example.test/data", archive, 1024)

    def test_empty_source_writes_failed_readiness(self):
        raw = self.root / "empty"
        raw.mkdir()
        prepared = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "Source inventory failed"):
            prepare(raw, prepared, self.config)
        self.assertFalse(read_json(prepared / "readiness.json")["verification_passed"])


if __name__ == "__main__":
    unittest.main()
