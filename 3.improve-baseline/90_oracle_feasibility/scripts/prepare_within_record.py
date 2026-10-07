"""Explicit exploratory time-block protocol; never substitutes for held-out recordings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, inside, load_config, now, rank, sha256, write_csv, write_json
from prepare_data import VIEWS, inventory, roles, time_block_bounds, validate_plan
from wisig_dataset import normalize_iq


def prepare_within_record(raw, output, config, fold=0):
    if config["split"]["unit"] != "time_block":
        raise ValueError("This preparer requires the explicit exploratory time-block configuration.")
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    records, errors = inventory(raw)
    synthetic = (raw / "SYNTHETIC_FIXTURE.json").is_file()
    labels = sorted({record["iq_configuration"] for record in records})
    write_json(output / "inventory.json", {"created_at_jst": now(), "raw_root": str(raw),
               "recordings": len(records), "classes": labels, "errors": errors, "synthetic_fixture": synthetic})
    try:
        if errors or len(labels) != config["split"]["expected_classes"]:
            raise ValueError(f"Inventory invalid: {len(labels)} configurations, {len(errors)} errors.")
        write_csv(output / "recordings.csv", records)
        class_roles = roles(config, labels, fold)
        class_map = {label: index for index, label in enumerate(class_roles["known"])}
        views, bounds = {view: [] for view in VIEWS}, {}
        for record in records:
            label = record["iq_configuration"]
            bounds[record["recording_id"]] = time_block_bounds(record["sample_count"], config)
            wave = np.memmap(inside(raw, record["data_path"]), dtype="<c16", mode="r")
            for population, (lower, upper) in bounds[record["recording_id"]].items():
                if label in class_map:
                    view = "train" if population == "train" else f"{population}_known"
                    numerical_label = class_map[label]
                elif population == "validation" and label in class_roles["validation_unknown"]:
                    view, numerical_label = "validation_unknown", -1
                elif population == "test" and label in class_roles["test_unknown"]:
                    view, numerical_label = "test_unknown", -1
                else:
                    continue
                possible = (upper - lower - 256) // config["input"]["stride"] + 1
                count = min(possible, config["split"]["max_windows_per_block"][population])
                starts = lower + np.linspace(0, possible - 1, count, dtype=np.int64) * config["input"]["stride"]
                windows = np.stack([wave[int(start):int(start) + 256] for start in starts])
                iq = normalize_iq(np.stack([windows.real, windows.imag], axis=-1), "sample_rms")
                relative = f"windows/{record['recording_id']}_{population}.npz"
                path = output / relative
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, iq=iq, sample_starts=starts)
                digest = sha256(path)
                for index, start in enumerate(starts):
                    views[view].append({"prepared_path": relative, "prepared_sha256": digest,
                        "window_index": index, "sample_start": int(start), "label": numerical_label,
                        "iq_configuration": label, "tx_id": record["tx_id"], "rx_id": "B210_fixed",
                        "run": record["run"], "split_group": record["split_group"],
                        "recording_id": record["recording_id"], "raw_sha256": record["raw_sha256"]})
        for view, rows in views.items():
            chosen = []
            for label in sorted({row["iq_configuration"] for row in rows}):
                population = [row for row in rows if row["iq_configuration"] == label]
                chosen.extend(rank(population, config["split"]["partition_seed"], f"window_cap|{view}")
                              [:config["split"]["max_windows_per_class_per_view"]])
            views[view] = chosen
        plan = {"schema_version": "oracle_within_record_time_split_v1", "created_at_jst": now(),
                "fold": fold, "config": config, "synthetic_fixture": synthetic, "scope": config["scope"],
                "independent_recording_evaluation": False, "raw_root": str(raw), "records": records,
                "record_manifest_sha256": sha256(output / "recordings.csv"), "class_roles": class_roles,
                "class_map": class_map, "block_bounds": bounds, "views": views}
        write_json(output / "split.json", plan)
        readiness = validate_plan(output, plan)
        readiness["split_sha256"] = sha256(output / "split.json")
        write_json(output / "readiness.json", readiness)
        return readiness
    except Exception as error:
        write_json(output / "readiness.json", {"verification_passed": False, "checked_at_jst": now(),
                   "error": f"{type(error).__name__}: {error}", "synthetic_fixture": synthetic})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oracle_iq_within_record.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/downloads/KRI-16IQImbalances-DemodulatedData")
    parser.add_argument("--output", type=Path, default=ROOT / "output/prepared/oracle_within_fold00")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.fold not in config["planned_folds"]:
        parser.error("Fold must be predeclared.")
    print(prepare_within_record(args.raw_root.resolve(), args.output.resolve(), config, args.fold))


if __name__ == "__main__":
    main()
