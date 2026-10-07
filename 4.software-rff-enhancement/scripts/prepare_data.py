"""Inventory ORACLE recordings and create source-group-disjoint I/Q windows."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import numpy as np

from common import ROOT, digest_json, inside, load_config, now, rank, read_json, sha256, write_csv, write_json
from wisig_dataset import normalize_iq

NAME = re.compile(r"^Demod_WiFi_cable_X310_(?P<tx>[^_]+)_IQ#?(?P<iq>\d+)_run(?P<run>\d+)$", re.I)
VIEWS = ("train", "validation_known", "validation_unknown", "test_known", "test_unknown")


def inventory(raw):
    records, errors = [], []
    data_hashes = {}
    for path in sorted(raw.rglob("*.sigmf-data")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        match = NAME.fullmatch(path.stem)
        meta = path.with_suffix(".sigmf-meta")
        if not match or not meta.is_file():
            errors.append({"path": str(path.relative_to(raw)), "error": "Unsupported name or missing matching .sigmf-meta"})
            continue
        try:
            document = read_json(meta)
            metadata = document.get("_metadata", document)
            if not isinstance(metadata.get("global"), dict):
                raise ValueError("Missing global SigMF object.")
            data_sha256, data_sha512 = hashlib.sha256(), hashlib.sha512()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    data_sha256.update(chunk)
                    data_sha512.update(chunk)
            digest = data_sha256.hexdigest()
            expected = metadata["global"].get("core:sha512")
            if expected and expected.lower() != data_sha512.hexdigest():
                raise ValueError("Publisher core:sha512 mismatch.")
            size = path.stat().st_size
            if size % 16 or size < 256 * 16:
                raise ValueError("Binary length is incompatible with complex128/256-point windows.")
            if digest in data_hashes:
                raise ValueError(f"Duplicate raw content: {data_hashes[digest]}")
            data_hashes[digest] = str(path.relative_to(raw))
            recording = digest_json([str(path.relative_to(raw)), digest])[:20]
            records.append({"recording_id": recording, "data_path": str(path.relative_to(raw)).replace("\\", "/"),
                            "metadata_path": str(meta.relative_to(raw)).replace("\\", "/"),
                            "raw_sha256": digest, "metadata_sha256": sha256(meta),
                            "publisher_sha512_verified": bool(expected), "bytes": size, "sample_count": size // 16,
                            "declared_dtype": metadata["global"].get("core:datatype", "missing"),
                            "metadata_layout": "legacy_wrapper" if "_metadata" in document else "standard",
                            "sample_rate_hz": metadata["global"].get("core:sample_rate"),
                            "effective_dtype": "<c16", "tx_id": match["tx"],
                            "iq_configuration": f"IQ{int(match['iq']):02d}", "run": int(match["run"]),
                            "split_group": f"{match['tx']}|run{int(match['run'])}"})
        except (ValueError, OSError, KeyError) as error:
            errors.append({"path": str(path.relative_to(raw)), "error": str(error)})
    return records, errors


def roles(config, labels, fold):
    labels = rank(labels, config["split"]["partition_seed"], "class_roles")
    n_test = config["split"]["test_unknown_classes"]
    rotated = labels[fold * n_test % len(labels):] + labels[:fold * n_test % len(labels)]
    n_validation = config["split"]["validation_unknown_classes"]
    result = {"test_unknown": rotated[:n_test], "validation_unknown": rotated[n_test:n_test + n_validation],
              "known": sorted(rotated[n_test + n_validation:])}
    return result


def partition_groups(config, records):
    ordered = rank(sorted({r["split_group"] for r in records}), config["split"]["partition_seed"], "source_groups")
    if len(ordered) < 3:
        raise ValueError("At least 3 independent Tx/run groups are required; window-level fallback is forbidden.")
    n_train = max(1, int(len(ordered) * config["split"]["fractions"][0]))
    n_validation = max(1, int(len(ordered) * config["split"]["fractions"][1]))
    n_train = min(n_train, len(ordered) - n_validation - 1)
    return {group: "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
            for i, group in enumerate(ordered)}


def validate_plan(prepared, plan):
    within = plan.get("schema_version") == "oracle_within_record_time_split_v1"
    if not within and plan.get("schema_version") != "oracle_window_split_v1":
        raise ValueError("Unsupported prepared split schema.")
    if within and (plan["config"]["split"]["unit"] != "time_block" or
                   plan["scope"] != "exploratory_within_record_injected_configuration_not_cross_day" or
                   plan.get("independent_recording_evaluation") is not False):
        raise ValueError("Time-block splits must explicitly retain their exploratory scope.")
    if sha256(prepared / "recordings.csv") != plan["record_manifest_sha256"]:
        raise ValueError("Recording manifest checksum differs from split.")
    expected_views = set(VIEWS)
    if set(plan["views"]) != expected_views:
        raise ValueError("Required views are missing.")
    groups = {"train": set(), "validation": set(), "test": set()}
    hashes = {"train": set(), "validation": set(), "test": set()}
    indices = set()
    class_roles = plan["class_roles"]
    if any(set(class_roles[a]) & set(class_roles[b]) for a, b in
           [("known", "validation_unknown"), ("known", "test_unknown"), ("validation_unknown", "test_unknown")]):
        raise ValueError("Class roles overlap.")
    cache = {}
    prepared_hashes = {}
    intervals = {}
    class_map = plan["class_map"]
    recordings = {r["recording_id"]: r for r in plan["records"]}
    if sorted(class_map.values()) != list(range(len(class_map))) or set(class_map) != set(class_roles["known"]):
        raise ValueError("Known class map is invalid.")
    for view, rows in plan["views"].items():
        if not rows:
            raise ValueError(f"Empty view: {view}")
        role = "validation_unknown" if view == "validation_unknown" else "test_unknown" if view == "test_unknown" else "known"
        population = "train" if view == "train" else "validation" if view.startswith("validation") else "test"
        if set(r["iq_configuration"] for r in rows) != set(class_roles[role]):
            raise ValueError(f"Incomplete class coverage in {view}.")
        for row in rows:
            source = recordings.get(row["recording_id"])
            if source is None or any(row[key] != source[key] for key in
                                     ["raw_sha256", "split_group", "iq_configuration", "tx_id", "run"]):
                raise ValueError("Source leakage or provenance mismatch in a window reference.")
            if within:
                bounds = time_block_bounds(source["sample_count"], plan["config"])
                if plan["block_bounds"][row["recording_id"]] != bounds:
                    raise ValueError("Time-block bounds differ from the predeclared fractions and gap.")
                lower, upper = bounds[population]
                if row["sample_start"] < lower or row["sample_start"] + 256 > upper:
                    raise ValueError("Window leaves its assigned time block or enters the guard gap.")
            elif plan["group_assignments"].get(row["split_group"]) != population:
                raise ValueError("Source leakage: window assigned to the wrong population.")
            key = (row["raw_sha256"], row["sample_start"])
            if key in indices:
                raise ValueError("A source window appears in multiple views.")
            indices.add(key)
            groups[population].add(row["split_group"])
            hashes[population].add(row["raw_sha256"])
            if row["label"] != (class_map[row["iq_configuration"]] if role == "known" else -1):
                raise ValueError("Window label differs from class role.")
            relative = row["prepared_path"]
            if relative not in cache:
                path = inside(prepared, relative)
                if sha256(path) != row["prepared_sha256"]:
                    raise ValueError("Prepared file SHA-256 mismatch.")
                with np.load(path, allow_pickle=False) as data:
                    iq, starts = data["iq"], data["sample_starts"]
                if iq.dtype != np.float32 or iq.shape[1:] != (256, 2) or not np.isfinite(iq).all():
                    raise ValueError("Invalid prepared I/Q arrays.")
                rms = np.sqrt(np.mean(np.sum(iq.astype(np.float64)**2, axis=-1), axis=-1))
                if not np.allclose(rms, 1, atol=1e-6):
                    raise ValueError("Prepared signal RMS differs from one.")
                cache[relative] = starts
                prepared_hashes[relative] = row["prepared_sha256"]
            elif prepared_hashes[relative] != row["prepared_sha256"]:
                raise ValueError("Conflicting prepared checksums in split.")
            starts = cache[relative]
            if row["window_index"] < 0 or row["window_index"] >= len(starts) or int(starts[row["window_index"]]) != row["sample_start"]:
                raise ValueError("Window index/start mismatch.")
            if row["sample_start"] < 0 or row["sample_start"] + 256 > source["sample_count"]:
                raise ValueError("Window leaves the raw recording.")
            intervals.setdefault(row["raw_sha256"], []).append(row["sample_start"])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not within and (groups[a] & groups[b] or hashes[a] & hashes[b]):
            raise ValueError(f"Source leakage between {a} and {b}.")
    for starts in intervals.values():
        if np.any(np.diff(sorted(starts)) < 256):
            raise ValueError("Source windows overlap.")
    return {"verification_passed": True, "checked_at_jst": now(), "synthetic_fixture": plan["synthetic_fixture"],
            "view_counts": {view: len(rows) for view, rows in plan["views"].items()},
            "source_groups": {view: len(value) for view, value in groups.items()},
            "cross_partition_source_overlap": len((hashes["train"] & hashes["validation"]) |
                                                   (hashes["train"] & hashes["test"]) |
                                                   (hashes["validation"] & hashes["test"])),
            "independent_recording_evaluation": not within, "overlapping_windows": 0,
            "scope": plan["scope"]}


def time_block_bounds(sample_count, config):
    split = config["split"]
    windows = sample_count // 256
    first = int(windows * split["fractions"][0])
    second = int(windows * sum(split["fractions"][:2]))
    gap = split["guard_windows_each_side"]
    bounds = {"train": [0, (first - gap) * 256],
              "validation": [(first + gap) * 256, (second - gap) * 256],
              "test": [(second + gap) * 256, windows * 256]}
    if gap < 1 or any(upper - lower < 256 for lower, upper in bounds.values()):
        raise ValueError("Recording too short for the declared time blocks and guard gaps.")
    return bounds


def prepare(raw, output, config, fold=0):
    if config["split"]["unit"] != "tx_run":
        raise ValueError("The strict preparer only supports Tx/run-disjoint evaluation; use the explicit exploratory preparer.")
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new output directory: {output}")
    output.mkdir(parents=True)
    records, errors = inventory(raw)
    labels = sorted({r["iq_configuration"] for r in records})
    synthetic = (raw / "SYNTHETIC_FIXTURE.json").is_file()
    write_json(output / "inventory.json", {"created_at_jst": now(), "raw_root": str(raw),
               "recordings": len(records), "classes": labels, "errors": errors, "synthetic_fixture": synthetic})
    if records:
        write_csv(output / "recordings.csv", records)
    try:
        if errors or not records:
            raise ValueError(f"Source inventory failed: {len(records)} recordings, {len(errors)} errors.")
        if len(labels) != config["split"]["expected_classes"]:
            raise ValueError(f"Expected {config['split']['expected_classes']} configurations, found {len(labels)}.")
        class_roles = roles(config, labels, fold)
        assignments = partition_groups(config, records)
        class_map = {label: i for i, label in enumerate(class_roles["known"])}
        views = {key: [] for key in VIEWS}
        for record in records:
            label, population = record["iq_configuration"], assignments[record["split_group"]]
            if label in class_roles["known"]:
                view = "train" if population == "train" else f"{population}_known"
                numerical_label = class_map[label]
            elif label in class_roles["validation_unknown"] and population == "validation":
                view, numerical_label = "validation_unknown", -1
            elif label in class_roles["test_unknown"] and population == "test":
                view, numerical_label = "test_unknown", -1
            else:
                continue
            wave = np.memmap(inside(raw, record["data_path"]), dtype="<c16", mode="r")
            possible = (len(wave) - 256) // config["input"]["stride"] + 1
            n = min(possible, config["input"]["max_windows_per_recording"])
            starts = np.linspace(0, possible - 1, n, dtype=np.int64) * config["input"]["stride"]
            windows = np.stack([wave[int(start):int(start) + 256] for start in starts])
            iq = normalize_iq(np.stack([windows.real, windows.imag], axis=-1), "sample_rms")
            relative = f"windows/{record['recording_id']}.npz"
            path = output / relative
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, iq=iq, sample_starts=starts)
            prepared_hash = sha256(path)
            for index, start in enumerate(starts):
                views[view].append({"prepared_path": relative, "prepared_sha256": prepared_hash,
                                    "window_index": index, "sample_start": int(start), "label": numerical_label,
                                    "iq_configuration": label, "tx_id": record["tx_id"], "rx_id": "B210_fixed",
                                    "run": record["run"], "split_group": record["split_group"],
                                    "recording_id": record["recording_id"], "raw_sha256": record["raw_sha256"]})
        cap = config["split"]["max_windows_per_class_per_view"]
        for view, rows in views.items():
            chosen = []
            for label in sorted({r["iq_configuration"] for r in rows}):
                population = [r for r in rows if r["iq_configuration"] == label]
                chosen.extend(rank(population, config["split"]["partition_seed"], f"window_cap|{view}")[:cap])
            views[view] = chosen
        plan = {"schema_version": "oracle_window_split_v1", "created_at_jst": now(), "fold": fold,
                "config": config, "synthetic_fixture": synthetic, "scope": config["scope"],
                "raw_root": str(raw), "records": records, "record_manifest_sha256": sha256(output / "recordings.csv"),
                "class_roles": class_roles, "class_map": class_map, "group_assignments": assignments, "views": views}
        write_json(output / "split.json", plan)
        readiness = validate_plan(output, plan)
        readiness["split_sha256"] = sha256(output / "split.json")
        write_json(output / "readiness.json", readiness)
        return readiness
    except Exception as error:
        write_json(output / "readiness.json", {"verification_passed": False, "checked_at_jst": now(),
                   "synthetic_fixture": synthetic, "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oracle_iq.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/oracle_dataset2")
    parser.add_argument("--output", type=Path, default=ROOT / "output/prepared/oracle_fold00")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.fold not in config["planned_folds"]:
        parser.error("Fold must be listed in planned_folds.")
    if args.validate_only:
        plan = read_json(args.output / "split.json")
        result = validate_plan(args.output, plan)
    elif args.inventory_only:
        records, errors = inventory(args.raw_root.resolve())
        result = {"recordings": len(records), "classes": sorted({r['iq_configuration'] for r in records}),
                  "groups": len({r['split_group'] for r in records}), "errors": errors}
        write_json(ROOT / "output/inventory.json", result)
        if errors or not records:
            print(result)
            raise SystemExit(2)
    else:
        result = prepare(args.raw_root.resolve(), args.output.resolve(), config, args.fold)
    print(result)


if __name__ == "__main__":
    main()
