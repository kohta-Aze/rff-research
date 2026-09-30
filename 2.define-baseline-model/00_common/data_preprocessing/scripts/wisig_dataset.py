"""NPZと共通分割を読むNumPyローダー。PyTorch等への変換はモデル側の担当。"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from wisig_common import sha256_file
from wisig_coverage import build_profile
from wisig_protocol import ranked, safe_signal_path, validate_fold


def normalize_iq(iq: np.ndarray, mode: str) -> np.ndarray:
    """I/Qをfloat32に統一。信号単位RMS、DC除去なし。ゼロ電力は明示的に停止。"""
    value = np.asarray(iq)
    if value.ndim not in (2, 3) or value.shape[-2:] != (256, 2):
        raise ValueError('信号の形は (256, 2) または (N, 256, 2) が必要。')
    if not np.isfinite(value).all():
        raise ValueError('信号に非有限値がある。')
    if mode == 'sample_rms':
        work = value.astype(np.float64)
        rms = np.sqrt(np.mean(np.sum(np.square(work), axis=-1), axis=-1))
        if np.any(rms <= 0) or not np.isfinite(rms).all():
            raise ValueError('RMS正規化できないゼロ電力・異常値。')
        value = work / rms[..., None, None]
    elif mode != 'none':
        raise ValueError(f'未対応の正規化: {mode}')
    with np.errstate(over='ignore'):
        result = np.array(value, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ValueError('float32変換で非有限値が発生。')
    return result


class WisigDataset:
    """1件ごとに (iq, label, metadata) を返す。未知ラベルは -1。"""

    def __init__(self, prepared_root: str | Path, split_file: str | Path, view: str,
                 *, cache_groups: int = 64):
        self.prepared_root = Path(prepared_root).resolve()
        self.plan = json.loads(Path(split_file).read_text(encoding='utf-8'))
        rows, report = build_profile(self.prepared_root)
        validate_fold(self.plan, rows, report)
        if view not in self.plan['views']:
            raise ValueError(f'分割に存在しないview: {view}')
        if type(cache_groups) is not int or cache_groups < 1:
            raise ValueError('cache_groups は正の整数が必要。')
        self.view = view
        self.normalization = self.plan['protocol']['normalization']
        self.class_map = dict(self.plan['class_map'])
        self.groups = self.plan['views'][view]
        self.references = [(g, i) for g, group in enumerate(self.groups) for i in group['signal_indices']]
        self.cache_groups = cache_groups
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.references)

    def _load_group(self, group: dict) -> np.ndarray:
        relative = group['relative_path']
        if relative not in self._cache:
            path = safe_signal_path(self.prepared_root, relative)
            if sha256_file(path) != group['file_sha256']:
                raise ValueError(f'NPZのハッシュ不一致: {relative}')
            with np.load(path, allow_pickle=False) as payload:
                if payload.files != ['iq']:
                    raise ValueError(f'NPZキーが不正: {relative}')
                iq = payload['iq']
            if iq.shape != (group['signal_count'], 256, 2) or str(iq.dtype) != group['dtype'] or not np.isfinite(iq).all():
                raise ValueError(f'NPZの形・型・値が不正: {relative}')
            iq.setflags(write=False)
            self._cache[relative] = iq
            if len(self._cache) > self.cache_groups:
                self._cache.popitem(last=False)
        self._cache.move_to_end(relative)
        return self._cache[relative]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, dict]:
        group_index, signal_index = self.references[index]
        group = self.groups[group_index]
        sample = normalize_iq(self._load_group(group)[signal_index], self.normalization)
        metadata = {key: group[key] for key in ('tx_id', 'rx_id', 'capture_date', 'representation', 'tx_role', 'relative_path')}
        metadata.update({'signal_index': signal_index, 'is_known': group['label'] >= 0,
                         'fold': self.plan['fold'], 'view': self.view, 'normalization': self.normalization})
        return sample, group['label'], metadata

    def iter_batches(self, batch_size: int, *, shuffle_seed: int | None = None, drop_last: bool = False):
        """(float32[B,256,2], int64[B], メタデータ一覧) を逐次返す。"""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size は正の整数が必要。')
        if shuffle_seed is not None and type(shuffle_seed) is not int:
            raise ValueError('shuffle_seed は整数が必要。')
        order = list(range(len(self)))
        if shuffle_seed is not None:
            order = ranked(order, shuffle_seed, f'batch|{self.plan["fold"]}|{self.view}')
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if drop_last and len(indices) < batch_size:
                break
            samples = [self[index] for index in indices]
            yield (np.stack([item[0] for item in samples]),
                   np.asarray([item[1] for item in samples], dtype=np.int64),
                   [item[2] for item in samples])
