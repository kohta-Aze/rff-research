"""Paths, provenance, and strict input helpers for the software-RFF experiment."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BASELINE = REPO / "2.define-baseline-model"
sys.path.insert(0, str(BASELINE / "00_common/modeling/scripts"))
sys.path.insert(0, str(BASELINE / "00_common/data_preprocessing/scripts"))
sys.path.insert(0, str(BASELINE / "01_cnn_msp/scripts"))
JST = timezone(timedelta(hours=9))


def now():
    return datetime.now(JST).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty manifest.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rank(values, seed, namespace):
    return sorted(values, key=lambda value: digest_json([seed, namespace, value]))


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves the dataset directory: {relative}")
    return path


def load_config(path):
    config = read_json(path)
    if config.get("schema_version") != "software_rff_v1" or config.get("dataset") != "oracle_dataset2":
        raise ValueError("Only the ORACLE Dataset #2 protocol is implemented.")
    inp, split = config["input"], config["split"]
    if inp["window_length"] != 256 or inp["stride"] < 256 or inp["complex_dtype"] != "<c16":
        raise ValueError("ORACLE requires non-overlapping 256-point complex128 windows.")
    if inp["normalization"] != "sample_rms" or inp["max_windows_per_recording"] < 1:
        raise ValueError("Unsupported normalization or window limit.")
    if split["unit"] != "tx_run" or len(split["fractions"]) != 3:
        raise ValueError("Split must group all IQ configurations from the same Tx/run.")
    if any(x <= 0 for x in split["fractions"]) or abs(sum(split["fractions"]) - 1) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to one.")
    counts = [split[k] for k in ["known_classes", "validation_unknown_classes", "test_unknown_classes"]]
    if counts[0] < 2 or min(counts[1:]) < 1 or sum(counts) != split["expected_classes"]:
        raise ValueError("Invalid class roles.")
    return config
