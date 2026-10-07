"""Run leakage-checked one-, two-, and three-day CNN + OpenMax experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "2.define-baseline-model").is_dir())
PREP = ROOT / "2.define-baseline-model/00_common/data_preprocessing"
MODEL = ROOT / "2.define-baseline-model"
sys.path.insert(0, str(MODEL / "01_cnn_msp/scripts"))
sys.path.insert(0, str(MODEL / "00_common/modeling/scripts"))
sys.path.insert(0, str(PREP / "scripts"))

import run_experiment as msp
from cnn_backbone import IQClassifier
from open_set_metrics import evaluate_open_set
from openmax import OpenMax, openmax_scores, select_openmax_threshold
from wisig_common import sha256_file
from wisig_coverage import build_profile
from wisig_dataset import normalize_iq
from wisig_protocol import ranked, validate_fold

JST = timezone(timedelta(hours=9))
DAYS = ("2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23")
CONDITIONS = {"A1_1day": (DAYS[:1], (120,)),
              "A2_2day": (DAYS[:2], (60, 60)),
              "A3_3day": (DAYS[:3], (40, 40, 40))}
VAL_PER_DAY = {"A1_1day": (40,), "A2_2day": (20, 20),
               "A3_3day": (13, 13, 14)}


def stable_digest(rows):
    value = [{"tx": row["tx_id"], "rx": row["rx_id"], "date": row["capture_date"],
              "index": int(row["signal_index"])} for row in rows]
    value.sort(key=lambda r: (r["date"], r["tx"], r["rx"], r["index"]))
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class Samples:
    def __init__(self, root, protocol, class_map):
        self.root = root
        self.protocol = protocol
        self.class_map = class_map
        self.cache = {}

    def load(self, rows, date, indices_by_group, role_filter=None):
        xs, ys, metadata, ids = [], [], [], []
        selected_groups = [row for row in rows if row["capture_date"] == date
                           and (role_filter is None or row["tx_role"] in role_filter)]
        for row in sorted(selected_groups, key=lambda r: (r["tx_id"], r["rx_id"])):
            group_id = (row["tx_id"], row["rx_id"], date)
            indices = indices_by_group.get(group_id, [])
            if not indices:
                continue
            rel = row["relative_path"]
            if rel not in self.cache:
                path = self.root / Path(*rel.replace("\\", "/").split("/"))
                if sha256_file(path) != row["file_sha256"]:
                    raise ValueError(f"Input hash mismatch: {rel}")
                with np.load(path, allow_pickle=False) as payload:
                    if payload.files != ["iq"]:
                        raise ValueError(f"Unexpected NPZ keys: {rel}")
                    data = payload["iq"]
                if data.shape != (row["signal_count"], 256, 2) or str(data.dtype) != row["dtype"]:
                    raise ValueError(f"Unexpected NPZ shape or dtype: {rel}")
                self.cache[rel] = data
            data = normalize_iq(self.cache[rel][indices], self.protocol["normalization"])
            xs.append(data)
            label = self.class_map.get(row["tx_id"], -1)
            ys.extend([label] * len(indices))
            for index in indices:
                ids.append({"tx_id": row["tx_id"], "rx_id": row["rx_id"],
                            "capture_date": date, "signal_index": int(index),
                            "relative_path": rel})
        if not xs:
            raise ValueError(f"No selected signals for {date}")
        iq = torch.from_numpy(np.ascontiguousarray(np.concatenate(xs).transpose(0, 2, 1)))
        labels = np.asarray(ys, dtype=np.int64)
        if len(labels) != len(ids) or not np.isfinite(iq.numpy()).all():
            raise ValueError("Selected sample integrity check failed.")
        return iq, labels, ids


def make_source_rows(coverage_rows, plan):
    known = set(plan["tx_roles"]["known"])
    val_unknown = set(plan["tx_roles"]["validation_unknown"])
    receivers = set(plan["protocol"]["primary_rx_ids"])
    result = []
    for row in coverage_rows:
        if row["rx_id"] not in receivers or row["tx_id"] not in known | val_unknown:
            continue
        if row["signal_count"] < 200:
            continue
        role = "known" if row["tx_id"] in known else "validation_unknown"
        result.append({**row, "tx_role": role})
    return result


def make_indices(plan, source_rows, source_days, train_counts, val_counts):
    train, validation = {}, {}
    test_refs = set()
    test_rows = {date: plan["views"][f"test_{date}"] for date in DAYS}
    for date in DAYS:
        for row in test_rows[date]:
            test_refs.update((row["relative_path"], int(i)) for i in row["signal_indices"])
    for date, n_train, n_val in zip(source_days, train_counts, val_counts):
        for row in source_rows:
            if row["capture_date"] != date:
                continue
            row_key = (row["tx_id"], row["rx_id"], date)
            order = ranked(range(row["signal_count"]), plan["protocol"]["split_seed"], row["relative_path"])
            test_ids = {i for rel, i in test_refs if rel == row["relative_path"]}
            pool = [i for i in order if i not in test_ids]
            # Known training uses the configured quota. Unknown Tx are validation-only.
            if row["tx_role"] == "known":
                train[row_key] = sorted(pool[:n_train])
                start = n_train
            else:
                start = 0
            validation[row_key] = sorted(pool[start:start + n_val])
            if len(validation[row_key]) != n_val or (row["tx_role"] == "known" and len(train[row_key]) != n_train):
                raise ValueError(f"Insufficient non-test samples: {row_key}")
    train_refs = set()
    for date in source_days:
        for row in source_rows:
            if row["capture_date"] != date:
                continue
            key = (row["tx_id"], row["rx_id"], date)
            if row["tx_role"] == "known":
                train_refs.update((row["relative_path"], i) for i in train.get(key, []))
            val_refs = {(row["relative_path"], i) for i in validation.get(key, [])}
            if val_refs & test_refs or val_refs & train_refs:
                raise ValueError("Training/validation/test sample overlap.")
    if train_refs & test_refs:
        raise ValueError("Training overlaps held-out test signals.")
    return train, validation


def source_indices(plan, date, max_per_group=120):
    """Candidate labelled calibration samples outside every frozen test split."""
    all_test = set()
    for day in DAYS:
        for row in plan["views"][f"test_{day}"]:
            all_test.update((row["relative_path"], int(i)) for i in row["signal_indices"])
    selected = {}
    for row in plan["views"][f"test_{date}"]:
        if row["tx_role"] != "known":
            continue
        order = ranked(range(row["signal_count"]), plan["protocol"]["split_seed"], row["relative_path"])
        pool = [i for i in order if (row["relative_path"], i) not in all_test]
        selected[(row["tx_id"], row["rx_id"], date)] = sorted(pool[:max_per_group])
    return selected


def select_calibration(labels, correct, metadata, dates, per_class, seed):
    chosen = []
    for label in sorted(set(labels.tolist())):
        for date, quota in zip(dates, per_class):
            candidates = [i for i, row in enumerate(metadata)
                          if labels[i] == label and correct[i] and row["capture_date"] == date]
            candidates = ranked(candidates, seed, f"calibration|{label}|{date}")
            if len(candidates) < quota:
                raise ValueError(f"Too few correct calibration signals: class={label}, day={date}, available={len(candidates)}")
            chosen.extend(candidates[:quota])
    return np.asarray(chosen, dtype=np.int64)


def infer_logits(model, iq, batch_size, device="cpu"):
    labels = np.zeros(len(iq), dtype=np.int64)
    return msp.infer(model, iq, labels, batch_size, device)[0].numpy().astype(np.float64)


def calibrate(train_logits, train_labels, val_logits, val_labels, config, class_count):
    best = None
    candidates = []
    for tail in config["tail_sizes"]:
        for alpha in config["alpha_ranks"]:
            calibration = OpenMax(class_count, tail, config["distance"], config["euclidean_scale"])
            calibration.fit(train_logits, train_labels)
            val_prob = calibration.probabilities(val_logits, alpha)
            val_pred, val_score, _ = openmax_scores(val_prob)
            selection = select_openmax_threshold(val_labels, val_pred, val_score)
            objective = selection["validation_balanced_open_set_accuracy"]
            candidates.append({"tail_size": tail, "alpha_rank": alpha, "validation_objective": objective,
                               "threshold": selection["threshold"]})
            if best is None or objective > best["objective"]:
                best = {"calibration": calibration, "alpha": alpha, "selection": selection,
                        "objective": objective, "tail": tail}
    return best, candidates


def evaluate(model, calibration, alpha, threshold, iq, labels):
    logits = infer_logits(model, iq, 64)
    probabilities = calibration.probabilities(logits, alpha)
    prediction, scores, _ = openmax_scores(probabilities)
    return evaluate_open_set(labels, prediction, scores, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "実験結果")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    out = args.output.resolve()
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = PREP / "output/prepared-wisig"
    base_split = PREP / "output/splits/day_shift_draft" / f"fold_{args.fold:02d}.json"
    readiness_path = PREP / "output/data_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("verification_passed"):
        raise ValueError("Source-data readiness is not verified.")
    rows, report = build_profile(prepared_root)
    plan = json.loads(base_split.read_text(encoding="utf-8"))
    validated = validate_fold(plan, rows, report)
    if not validated["verification_passed"]:
        raise ValueError("Base fold did not pass source split validation.")
    for date in DAYS:
        if f"test_{date}" not in plan["views"]:
            raise ValueError(f"Missing frozen test view: {date}")
    rx = plan["protocol"]["primary_rx_ids"]
    class_map = plan["class_map"]
    openmax_config = json.loads((MODEL / "03_cnn_openmax/experiments/openmax_baseline.json").read_text(encoding="utf-8"))
    msp_config = json.loads((MODEL / "01_cnn_msp/experiments/cnn_msp_baseline.json").read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cache = Samples(prepared_root, plan["protocol"], class_map)
    all_metrics = []
    fitted = {}
    start_all = time.perf_counter()

    for name, (source_days, daily_counts) in CONDITIONS.items():
        print(f"\n=== {name}: train={sum(daily_counts)} total, days={source_days} ===", flush=True)
        quotas = VAL_PER_DAY[name]
        day_train_counts = {day: n for day, n in zip(source_days, daily_counts)}
        source_rows = make_source_rows(rows, plan)
        train_map, val_map = make_indices(plan, source_rows, source_days, daily_counts, quotas)
        train_parts, val_known_parts, val_unknown_parts = [], [], []
        train_meta, val_meta = [], []
        for day in source_days:
            train_data = cache.load(source_rows, day, train_map, {"known"})
            val_known = cache.load(source_rows, day, val_map, {"known"})
            unknown_data = cache.load(source_rows, day, val_map, {"validation_unknown"})
            train_parts.append(train_data)
            val_known_parts.append(val_known)
            val_unknown_parts.append(unknown_data)
            train_meta.extend(train_data[2]); val_meta.extend(val_known[2])
            print(f"Loaded {day}: train={len(train_data[1])}, val-known={len(val_known[1])}, "
                  f"val-unknown={len(unknown_data[1])}", flush=True)
        train_data = (torch.cat([part[0] for part in train_parts]),
                      np.concatenate([part[1] for part in train_parts]), train_meta)
        val_known = (torch.cat([part[0] for part in val_known_parts]),
                     np.concatenate([part[1] for part in val_known_parts]), val_meta)
        val_unknown = (torch.cat([part[0] for part in val_unknown_parts]),
                       np.concatenate([part[1] for part in val_unknown_parts]),
                       [row for part in val_unknown_parts for row in part[2]])
        val_data = (torch.cat([val_known[0], val_unknown[0]]),
                    np.concatenate([val_known[1], val_unknown[1]]),
                    val_known[2] + val_unknown[2])
        if len(train_data[1]) != 11520 or len(val_data[1]) != 5120:
            raise ValueError("Train or validation total differs across conditions.")
        if np.any(train_data[1] < 0) or np.any(val_known[1] < 0) or not np.any(val_unknown[1] < 0):
            raise ValueError("Known and unknown split labels are invalid.")
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(args.seed)
        model = IQClassifier(len(class_map), **msp_config["architecture"])
        training = msp.train(model, train_data, val_known, msp_config, args.seed, torch.device("cpu"), run_dir)
        train_logits = infer_logits(model, train_data[0], msp_config["training"]["batch_size"])
        val_logits = infer_logits(model, val_data[0], openmax_config["inference_batch_size"])
        best, candidates = calibrate(train_logits, train_data[1], val_logits, val_data[1],
                                     openmax_config, len(class_map))
        (run_dir / "openmax_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
        day_results = {}
        for day in DAYS:
            test_rows = plan["views"][f"test_{day}"]
            test_iq, test_labels, _ = cache.load(test_rows, day,
                                                  {(r["tx_id"], r["rx_id"], day): r["signal_indices"]
                                                   for r in test_rows}, {"known", "test_unknown"})
            metrics = evaluate(model, best["calibration"], best["alpha"], best["selection"]["threshold"],
                               test_iq, test_labels)
            entry = {"condition": name, "test_date": day, **metrics}
            all_metrics.append(entry)
            day_results[day] = metrics
            print(f"{name} {day}: OSCR={metrics['oscr_auc']:.4f} AUROC={metrics['auroc_known_positive']:.4f} "
                  f"CCR={metrics['known_correct_accept_rate']:.3f} FAR={metrics['unknown_false_accept_rate']:.3f}", flush=True)
            del test_iq, test_labels
        run_info = {"condition": name, "source_days": list(source_days),
                    "train_per_tx_rx_day": dict(zip(source_days, daily_counts)),
                    "validation_per_tx_rx_day": dict(zip(source_days, quotas)),
                    "train_count": len(train_data[1]), "validation_known_count": len(val_known[1]),
                    "validation_unknown_count": len(val_unknown[1]), "train_indices_sha256": stable_digest(train_meta),
                    "validation_indices_sha256": stable_digest(val_meta + val_unknown[2]),
                    "checkpoint_sha256": sha256_file(run_dir / "best_model.pt"),
                    "training": training, "selected_tail_size": best["tail"],
                    "selected_alpha_rank": best["alpha"], "threshold": best["selection"]["threshold"],
                    "test": day_results}
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        fitted[name] = {"model": model, "train_data": train_data, "train_logits": train_logits,
                        "train_meta": train_meta, "validation": (val_data, val_logits),
                        "best": best}
        del train_parts, val_known_parts, val_unknown_parts, train_data, val_data, val_known, val_unknown

    # H2: keep the A1 CNN fixed; hold calibration sample count at 100/class.
    source = fitted["A1_1day"]
    model = source["model"]
    model.requires_grad_(False)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    calib_conditions = [
        ("H2_single_day", (DAYS[0],), (100,)),
        ("H2_pooled_three_days", DAYS[:3], (34, 33, 33)),
    ]
    h2_results = []
    for label, days, quotas in calib_conditions:
        logits_parts, labels_parts, meta_parts = [], [], []
        for day, quota in zip(days, quotas):
            indices = source_indices(plan, day, max_per_group=120)
            candidate = cache.load(source_rows, day, indices, {"known"})
            logits = infer_logits(model, candidate[0], 64)
            correct = logits.argmax(axis=1) == candidate[1]
            keep = select_calibration(candidate[1], correct, candidate[2], (day,), (quota,), args.seed)
            logits_parts.append(logits[keep]); labels_parts.append(candidate[1][keep])
            meta_parts.extend(candidate[2][i] for i in keep)
        fit_logits = np.concatenate(logits_parts)
        fit_labels = np.concatenate(labels_parts)
        (validation, validation_logits) = source["validation"]
        best, candidates = calibrate(fit_logits, fit_labels, validation_logits, validation[1],
                                     openmax_config, len(class_map))
        test = {}
        for day in DAYS:
            test_rows = plan["views"][f"test_{day}"]
            test_iq, test_labels, _ = cache.load(test_rows, day,
                                                  {(r["tx_id"], r["rx_id"], day): r["signal_indices"]
                                                   for r in test_rows}, {"known", "test_unknown"})
            metrics = evaluate(model, best["calibration"], best["alpha"],
                               best["selection"]["threshold"], test_iq, test_labels)
            test[day] = metrics
            h2_results.append({"condition": label, "test_date": day, **metrics})
            print(f"{label} {day}: OSCR={metrics['oscr_auc']:.4f} AUROC={metrics['auroc_known_positive']:.4f} "
                  f"CCR={metrics['known_correct_accept_rate']:.3f} FAR={metrics['unknown_false_accept_rate']:.3f}", flush=True)
        record = {"condition": label, "cnn_checkpoint_sha256": sha256_file(runs_dir / "A1_1day/best_model.pt"),
                  "calibration_days": list(days), "per_class_fit_count": 100,
                  "fit_total": int(len(fit_labels)), "fit_indices_sha256": stable_digest(meta_parts),
                  "selected_tail_size": best["tail"], "selected_alpha_rank": best["alpha"],
                  "threshold": best["selection"]["threshold"], "test": test}
        (runs_dir / f"{label}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (runs_dir / f"{label}_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in state.items()):
        raise ValueError("CNN weights changed during frozen calibration comparison.")

    all_metrics.extend(h2_results)
    metrics_path = out / "結果.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]))
        writer.writeheader(); writer.writerows(all_metrics)
    manifest = {"generated_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
                "fold": args.fold, "seed": args.seed, "source_commit": plan["provenance"],
                "base_split_sha256": sha256_file(base_split), "readiness_sha256": sha256_file(readiness_path),
                "conditions": CONDITIONS, "validation_quotas": VAL_PER_DAY,
                "n_rx": len(rx), "n_known_tx": len(class_map), "threads": args.threads,
                "runtime_seconds": time.perf_counter() - start_all,
                "scope": "exploratory_single_fold_single_seed",
                "test_signal_refs_sha256": {day: stable_digest([
                    {"tx_id": row["tx_id"], "rx_id": row["rx_id"], "capture_date": day,
                     "signal_index": i, "relative_path": row["relative_path"]}
                    for row in plan["views"][f"test_{day}"] for i in row["signal_indices"]]) for day in DAYS}}
    (out / "実験条件.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nCompleted: {metrics_path} ({manifest['runtime_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
