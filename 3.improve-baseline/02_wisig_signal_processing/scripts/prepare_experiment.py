"""Pin the selected baseline and verify its existing data/split provenance."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paths import BASELINE, CONTRACT, PREP, REFERENCE, REPO, read_json, sha256, write_json
from wisig_coverage import build_profile
from wisig_protocol import validate_fold


def prepare(output):
    if output.exists():
        raise FileExistsError('Contract exists; use verify_contract rather than replacing it.')
    run = read_json(REFERENCE / 'run.json')
    if run['status'] != 'completed' or run['method'] != 'CNN + OpenMax' or run['cnn_retrained']:
        raise ValueError('Reference must be the completed frozen-CNN OpenMax baseline.')
    split = PREP / f'output/splits/day_shift_draft/fold_{run["fold"]:02d}.json'
    expected_assets = {
        REFERENCE / 'cnn_model.pt': run['checkpoint_sha256'],
        REFERENCE / 'calibration.json': run['calibration_sha256'],
        REFERENCE / 'threshold.json': run['threshold_sha256'],
        split: run['split_sha256'],
    }
    for relative, digest in run['source_files'].items():
        expected_assets[BASELINE / relative] = digest
    for path, digest in expected_assets.items():
        if sha256(path) != digest:
            raise ValueError(f'Baseline provenance mismatch: {path}')
    plan = read_json(split)
    if plan['class_map'] != run['class_map'] or plan['tx_roles'] != run['tx_roles']:
        raise ValueError('Baseline transmitter roles changed.')
    rows, report = build_profile(PREP / 'output/prepared-wisig')
    validate_fold(plan, rows, report)
    for name in ['run.json', 'metrics.json']:
        path = REFERENCE / name
        expected_assets[path] = sha256(path)
    contract = {
        'schema_version': 'wisig_baseline_contract_v1',
        'created_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        'baseline_reference': REFERENCE.relative_to(REPO).as_posix(),
        'split': split.relative_to(REPO).as_posix(),
        'prepared_root': (PREP / 'output/prepared-wisig').relative_to(REPO).as_posix(),
        'fold': run['fold'], 'seed': run['seed'],
        'tx_roles': run['tx_roles'], 'class_map': run['class_map'],
        'test_status': 'previously_inspected_exploratory_dates',
        'file_sha256': {path.relative_to(REPO).as_posix(): digest for path, digest in expected_assets.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, contract)
    print(f'Baseline pinned: fold {run["fold"]}, seed {run["seed"]}, {len(expected_assets)} verified files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CONTRACT)
    args = parser.parse_args()
    prepare(args.output)
