"""Locate unchanged baseline assets; write only to this study."""
from pathlib import Path
import hashlib
import json
import sys

STUDY = Path(__file__).resolve().parents[1]
REPO = next(p for p in STUDY.parents if (p / '2.define-baseline-model').is_dir())
BASELINE = REPO / '2.define-baseline-model'
PREP = BASELINE / '00_common/data_preprocessing'
REFERENCE = BASELINE / '03_cnn_openmax/output/20261006_primary_fold00_seed42'
CONTRACT = STUDY / 'configs/baseline_contract.json'
for folder in ['00_common/modeling/scripts', '00_common/data_preprocessing/scripts', '01_cnn_msp/scripts']:
    sys.path.insert(0, str(BASELINE / folder))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repo_path(relative):
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError('Reference path leaves repository.')
    return path


def verify_contract(path=CONTRACT):
    contract = read_json(path)
    if contract['schema_version'] != 'wisig_baseline_contract_v1':
        raise ValueError('Unsupported baseline contract.')
    for relative, expected in contract['file_sha256'].items():
        if sha256(repo_path(relative)) != expected:
            raise ValueError(f'Baseline changed: {relative}')
    return contract
