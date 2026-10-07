"""Save the experiment matrix first, then prepare and run the requested jobs."""
import argparse
from pathlib import Path
import subprocess
import sys

from common import ROOT, load_config, now, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oracle_iq.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/oracle_dataset2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if not set(args.folds) <= set(config["planned_folds"]) or not set(args.seeds) <= set(config["planned_seeds"]):
        parser.error("Requested folds/seeds must be predeclared in the configuration.")
    if len(set(args.folds)) != len(args.folds) or len(set(args.seeds)) != len(args.seeds) or args.threads < 1:
        parser.error("Folds/seeds must be distinct and threads positive.")
    output = args.output or ROOT / "output/matrices" / now().replace(":", "").replace("-", "")[:15]
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    preparation_script = "prepare_within_record.py" if config["split"]["unit"] == "time_block" else "prepare_data.py"
    for fold in args.folds:
        prepared = output / f"prepared_fold{fold:02d}"
        preparation = [sys.executable, str(ROOT / "scripts" / preparation_script), "--config", str(args.config.resolve()),
                       "--raw-root", str(args.raw_root.resolve()), "--output", str(prepared.resolve()), "--fold", str(fold)]
        for seed in args.seeds:
            training = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--prepared", str(prepared.resolve()),
                        "--output", str((output / f"fold{fold:02d}_seed{seed}").resolve()), "--seed", str(seed), "--threads", str(args.threads)]
            if args.evaluate_test:
                training.extend(["--evaluate-test", "--awgn-db", "20", "10"])
            jobs.append({"fold": fold, "seed": seed, "prepare_command": preparation, "train_command": training})
    record = {"created_at_jst": now(), "config_sha256": sha256(args.config), "status": "dry_run" if args.dry_run else "running",
              "evaluate_test": args.evaluate_test, "jobs": jobs, "completed_jobs": [], "scope": config["scope"]}
    write_json(output / "matrix.json", record)
    if args.dry_run:
        print(f"Matrix saved, no training: {output / 'matrix.json'}")
        return
    try:
        prepared_folds = set()
        for job in jobs:
            if job["fold"] not in prepared_folds:
                subprocess.run(job["prepare_command"], check=True)
                prepared_folds.add(job["fold"])
            subprocess.run(job["train_command"], check=True)
            record["completed_jobs"].append({"fold": job["fold"], "seed": job["seed"]})
            write_json(output / "matrix.json", record)
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "matrix.json", record)


if __name__ == "__main__":
    main()
