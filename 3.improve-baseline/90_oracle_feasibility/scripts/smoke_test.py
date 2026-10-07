"""Exercise archive import, splitting, CNN and native OpenMax on marked fake data."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from common import ROOT, load_config, now, read_json, write_json
from prepare_data import prepare
from download_data import extract


def fixtures(raw):
    raw.mkdir(parents=True)
    write_json(raw / "SYNTHETIC_FIXTURE.json", {"synthetic": True, "purpose": "pipeline test only"})
    # Deliberately large, stationary differences make this a software check, not an RF benchmark.
    for tx in range(4):
        for run in range(3):
            for label in range(4):
                rng = np.random.default_rng(tx * 1000 + run * 100 + label)
                qpsk = rng.choice([-1., 1.], 96 * 256) + 1j * rng.choice([-1., 1.], 96 * 256)
                values = (qpsk.real * (1 + label * .8) + 1j * qpsk.imag * (1 - label * .15)
                          + label * (1.5 + .5j) + rng.normal(0, .02, len(qpsk)))
                stem = raw / f"Demod_WiFi_cable_X310_SYN{tx}_IQ#{label + 1}_run{run + 1}"
                values.astype("<c16").tofile(stem.with_suffix(".sigmf-data"))
                # Test the official metadata mismatch correction explicitly.
                write_json(stem.with_suffix(".sigmf-meta"), {"global": {"core:datatype": "cf32", "core:sample_rate": 5000000}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep outputs at this new directory.")
    args = parser.parse_args()
    out = args.output.resolve() if args.output else ROOT / "output" / f"smoke_{now().replace(':', '').replace('-', '')[:15]}"
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw"
    fixtures(raw)
    with zipfile.ZipFile(out / "fixture.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for path in raw.iterdir():
            archive.write(path, path.name)
    extract(out / "fixture.zip", out / "imported", 64 * 1024 * 1024)
    # The marker is included so an imported fixture cannot become a real-data result.
    raw = out / "imported"
    config = deepcopy(load_config(ROOT / "configs/oracle_iq.json"))
    config["split"].update(known_classes=2, validation_unknown_classes=1, test_unknown_classes=1,
                           expected_classes=4, max_windows_per_class_per_view=384)
    config["input"]["max_windows_per_recording"] = 64
    write_json(out / "smoke_config.json", config)
    readiness = prepare(raw, out / "prepared", config)
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--prepared", str(out / "prepared"),
               "--output", str(out / "run"), "--max-epochs", "6", "--threads", "2", "--allow-synthetic", "--evaluate-test",
               "--awgn-db", "20", "10"]
    subprocess.run(command, check=True)
    run = read_json(out / "run/run.json")
    if run["status"] != "completed" or not run["synthetic_fixture"] or run["scientific_result"]:
        raise ValueError("Smoke outputs must be completed synthetic checks, never scientific results.")
    with (out / "run/openmax/test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        columns = csv.DictReader(stream).fieldnames
    if "closed_set_predicted_configuration" not in columns or "tx_id" not in columns:
        raise ValueError("Export must distinguish injected configurations from physical Tx IDs.")
    write_json(out / "smoke_result.json", {"status": "passed", "checked_at_jst": now(), "synthetic_fixture": True,
               "scientific_result": False, "readiness": readiness, "run": str(out / "run"),
               "checks": ["ZIP import", "complex128 despite cf32 metadata", "256-point nonoverlapping windows", "Tx/run-group isolation",
                          "RMS normalization", "existing CNN training", "MSP threshold", "native libMR OpenMax", "score/curve export", "AWGN stress export"]})
    print(f"Smoke passed. These are synthetic software checks only: {out}")


if __name__ == "__main__":
    main()
