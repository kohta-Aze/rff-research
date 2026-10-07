"""Train the existing CNN and compare MSP/OpenMax on injected signatures."""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import importlib.util
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from common import BASELINE, REPO, ROOT, digest_json, inside, now, read_json, sha256, write_csv, write_json
from prepare_data import validate_plan
from cnn_backbone import IQClassifier
from open_set_metrics import select_threshold
from open_set_outputs import export_scores
from openmax import OpenMax, openmax_scores, select_openmax_threshold

# Explicit file loading avoids colliding with this runner's own module name.
spec = importlib.util.spec_from_file_location("baseline_cnn_training", BASELINE / "01_cnn_msp/scripts/run_experiment.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def load_view(prepared, rows):
    cache = {}
    signals = []
    for row in rows:
        key = row["prepared_path"]
        if key not in cache:
            with np.load(inside(prepared, key), allow_pickle=False) as archive:
                cache[key] = archive["iq"]
        signals.append(cache[key][row["window_index"]])
    iq = torch.from_numpy(np.ascontiguousarray(np.stack(signals).transpose(0, 2, 1)))
    return iq, np.array([r["label"] for r in rows], dtype=np.int64), rows


def join_views(a, b):
    return torch.cat([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2] + b[2]


def add_awgn(data, snr_db, seed):
    # RMS-normalized input has complex power one. No labels determine the transformation.
    from wisig_dataset import normalize_iq
    generator = np.random.default_rng(seed)
    signal = data[0].numpy().transpose(0, 2, 1).astype(np.float64)
    noise = generator.normal(0, np.sqrt(10 ** (-snr_db / 10) / 2), signal.shape)
    transformed = normalize_iq(signal + noise, "sample_rms")
    return torch.from_numpy(np.ascontiguousarray(transformed.transpose(0, 2, 1))), data[1], data[2]


def calibrate(train_logits, train_labels, validation_logits, validation_labels, settings, class_count):
    best, candidates = None, []
    for tail in settings["tail_sizes"]:
        try:
            fitted = OpenMax(class_count, tail, settings["distance"], settings["euclidean_scale"]).fit(train_logits, train_labels)
        except ValueError as error:
            candidates.append({"tail_size": tail, "status": "unavailable", "error": str(error)})
            continue
        for alpha in settings["alpha_ranks"]:
            if alpha > class_count:
                continue
            probabilities = fitted.probabilities(validation_logits, alpha)
            predictions, scores, _ = openmax_scores(probabilities)
            selection = select_openmax_threshold(validation_labels, predictions, scores)
            objective = selection["validation_balanced_open_set_accuracy"]
            candidates.append({"tail_size": tail, "alpha_rank": alpha, "status": "valid",
                               "validation_objective": objective, "threshold": selection["threshold"]})
            if best is None or objective > best["objective"]:
                best = {"calibration": fitted, "alpha": alpha, "selection": selection, "objective": objective}
    if best is None:
        raise ValueError("No valid OpenMax fit. Inspect training accuracy and per-class correctly classified counts.")
    return best, candidates


def export_pair(model, data, best, msp_selection, batch_size, out, view, class_map):
    logits, predictions, scores = baseline.infer(model, data[0], data[1], batch_size, "cpu")
    values = logits.numpy().astype(np.float64)
    results = {"cnn_msp": export_configuration_scores(out / "msp", view, data[1], predictions, scores, data[2],
               msp_selection["threshold"], class_map, arrays={"logits": values})}
    probability = best["calibration"].probabilities(values, best["alpha"])
    predictions, scores, unknown_wins = openmax_scores(probability)
    results["cnn_openmax"] = export_configuration_scores(out / "openmax", view, data[1], predictions, scores, data[2],
        best["selection"]["threshold"], class_map, arrays={"logits": values, "probabilities": probability,
                                                          "unknown_wins": unknown_wins})
    return results


def export_configuration_scores(output, view, *args, **kwargs):
    result = export_scores(output, view, *args, **kwargs)
    # Class labels identify injected settings; tx_id in source metadata remains the physical radio.
    rename = {"closed_set_predicted_tx": "closed_set_predicted_configuration",
              "open_set_predicted_tx": "open_set_predicted_configuration",
              "true_tx": "true_configuration", "predicted_tx": "predicted_configuration"}
    for suffix in ["predictions", "confusion"]:
        path = output / f"{view}_{suffix}.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [{rename.get(key, key): value for key, value in row.items()}
                    for row in csv.DictReader(stream)]
        write_csv(path, rows)
    return result


def run(args):
    if args.threads < 1 or args.seed < 0 or (args.max_epochs is not None and args.max_epochs < 1):
        raise ValueError("Invalid thread count, seed, or epoch count.")
    prepared = args.prepared.resolve()
    plan = read_json(prepared / "split.json")
    readiness = read_json(prepared / "readiness.json")
    if not readiness.get("verification_passed") or sha256(prepared / "split.json") != readiness["split_sha256"]:
        raise ValueError("Prepared readiness or split checksum is invalid.")
    validate_plan(prepared, plan)
    if plan["synthetic_fixture"] and not args.allow_synthetic:
        raise ValueError("Synthetic fixture requires --allow-synthetic and cannot supply a scientific result.")
    config = read_json(REPO / plan["config"]["cnn_config"])
    baseline.validate_config(config)
    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs
    openmax_config = read_json(REPO / plan["config"]["openmax_config"])
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    stamp = now().replace(":", "").replace("-", "")[:15]
    out = args.output.resolve() if args.output else ROOT / "output/runs" / f"{stamp}_fold{plan['fold']:02d}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    for folder in ["msp", "openmax", "source_snapshot"]:
        (out / folder).mkdir()
    source_files = [*sorted((ROOT / "scripts").glob("*.py")), Path(baseline.__file__),
                    BASELINE / "00_common/modeling/scripts/cnn_backbone.py",
                    BASELINE / "00_common/modeling/scripts/openmax.py",
                    BASELINE / "00_common/modeling/scripts/open_set_metrics.py",
                    BASELINE / "00_common/modeling/scripts/open_set_outputs.py",
                    BASELINE / "00_common/data_preprocessing/scripts/wisig_dataset.py"]
    sources = {}
    for source in source_files:
        relative = source.relative_to(REPO)
        snapshot = out / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        sources[str(relative)] = sha256(source)
    write_json(out / "cnn_config.json", config)
    write_json(out / "openmax_config.json", openmax_config)
    shutil.copy2(prepared / "split.json", out / "split.json")
    started = time.perf_counter()
    record = {"status": "running", "started_at_jst": now(), "fold": plan["fold"], "seed": args.seed,
              "synthetic_fixture": plan["synthetic_fixture"],
              "scientific_result": False,
              "scope": plan["scope"], "label_target": "injected_iq_configuration", "class_roles": plan["class_roles"],
              "class_map": plan["class_map"], "split_sha256": sha256(prepared / "split.json"),
              "runtime": {"python": sys.executable, "threads": args.threads, "device": "cpu",
                          "packages": {name: version(name) for name in ["numpy", "torch", "scipy", "scikit-learn", "libmr"]}},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "source_files": sources,
              "test_executed": False}
    write_json(out / "run.json", record)
    try:
        train = load_view(prepared, plan["views"]["train"])
        val_known = load_view(prepared, plan["views"]["validation_known"])
        model = IQClassifier(len(plan["class_map"]), **config["architecture"])
        record["training"] = baseline.train(model, train, val_known, config, args.seed, torch.device("cpu"), out)
        val_unknown = load_view(prepared, plan["views"]["validation_unknown"])
        validation = join_views(val_known, val_unknown)
        batch_size = config["training"]["batch_size"]
        train_logits = baseline.infer(model, train[0], train[1], batch_size, "cpu")[0].numpy().astype(np.float64)
        val_logits, predictions, scores = baseline.infer(model, validation[0], validation[1], batch_size, "cpu")
        msp_selection = select_threshold(validation[1], predictions, scores)
        best, candidates = calibrate(train_logits, train[1], val_logits.numpy().astype(np.float64),
                                     validation[1], openmax_config, len(plan["class_map"]))
        write_json(out / "openmax_candidates.json", candidates)
        write_json(out / "openmax/calibration.json", {**best["calibration"].state(), "alpha_rank": best["alpha"]})
        write_json(out / "openmax/threshold.json", best["selection"])
        write_json(out / "msp/threshold.json", msp_selection)
        record.update(status="validation_complete", checkpoint_sha256=sha256(out / "best_model.pt"),
                      selected_tail_size=best["calibration"].tail_size, selected_alpha_rank=best["alpha"],
                      calibration_sha256=sha256(out / "openmax/calibration.json"),
                      msp_threshold=msp_selection, openmax_threshold=best["selection"],
                      choices_saved_before_test=True)
        write_json(out / "run.json", record)
        metrics = {"validation": export_pair(model, validation, best, msp_selection, batch_size, out,
                                              "validation", plan["class_map"])}
        if args.evaluate_test:
            # Test data enter model inference only after all fitted choices have been saved.
            test = join_views(load_view(prepared, plan["views"]["test_known"]),
                              load_view(prepared, plan["views"]["test_unknown"]))
            metrics["test"] = export_pair(model, test, best, msp_selection, batch_size, out, "test", plan["class_map"])
            for snr_db in args.awgn_db:
                view = f"test_awgn_{snr_db:g}db"
                noisy_test = add_awgn(test, snr_db, args.seed + 10000)
                metrics[view] = export_pair(model, noisy_test, best, msp_selection, batch_size, out, view, plan["class_map"])
            record["test_executed"] = True
        write_json(out / "metrics.json", metrics)
        write_csv(out / "metrics.csv", [{"view": view, "method": method, "synthetic_fixture": plan["synthetic_fixture"], **value}
                  for view, methods in metrics.items() for method, value in methods.items()])
        record.update(status="completed", ended_at_jst=now(), elapsed_seconds=time.perf_counter() - started,
                      scientific_result=not plan["synthetic_fixture"] and args.evaluate_test,
                      awgn_db=args.awgn_db if args.evaluate_test else [])
        write_json(out / "run.json", record)
        print(f"Completed: {out}; synthetic_fixture={plan['synthetic_fixture']}; test_executed={args.evaluate_test}")
    except BaseException as error:
        record.update(status="failed", scientific_result=False, error=f"{type(error).__name__}: {error}", ended_at_jst=now())
        write_json(out / "run.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "output/prepared/oracle_fold00")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--awgn-db", type=float, nargs="*", default=[], help="Predeclared receive-side noise stress tests; requires --evaluate-test.")
    args = parser.parse_args()
    if args.awgn_db and (not args.evaluate_test or not np.isfinite(args.awgn_db).all() or len(set(args.awgn_db)) != len(args.awgn_db)):
        parser.error("AWGN tests require --evaluate-test and distinct finite SNR values.")
    run(args)


if __name__ == "__main__":
    main()
