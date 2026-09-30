"""合成波形によるデータ整合・分割漏洩・再生成・ローダーの回帰テスト。"""
from copy import deepcopy
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from wisig_common import sha256_file, summarize_array, write_json
from wisig_coverage import build_profile
from wisig_dataset import WisigDataset, normalize_iq
from wisig_protocol import allocation_rows, build_fold, ranked, safe_signal_path, validate_config, validate_fold


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_splits = script_module('04_build_splits').build_splits
verify_data = script_module('05_verify_data').verify_data
prepare = script_module('02_prepare_wisig')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.prepared = cls.folder / 'prepared'
        cls.prepared.mkdir()
        cls.source = cls.folder / 'source.pkl'
        cls.source.write_bytes(b'fixture source; not loaded as pickle')
        cls.config = json.loads((ROOT / 'configs' / 'day_shift_draft.json').read_text())
        cls.config['primary_rx_ids'] = ['rx_0', 'rx_1']
        cls.config_path = cls.folder / 'config.json'
        write_json(cls.config_path, cls.config)
        saved, skipped = [], []
        dates = cls.config['test_dates']
        for tx in range(10):
            for rx in range(3):
                for day, date in enumerate(dates):
                    count = 0 if (tx, rx, day) == (0, 2, 0) else (5 if (tx, rx, day) == (1, 2, 1) else 200)
                    info = {'tx_index': tx, 'tx_id': f'tx_{tx}', 'rx_index': rx, 'rx_id': f'rx_{rx}',
                            'day_index': day, 'capture_date': date, 'representation': 'equalized', 'signal_count': count}
                    if count == 0:
                        skipped.append({**info, 'shape': [0, 256, 2], 'reason': 'empty_source_group'})
                        continue
                    relative = f'signals/equalized/tx_{tx}__rx_{rx}__day_{day}.npz'
                    path = cls.prepared / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    iq = np.full((count, 256, 2), tx + rx + day + 1, dtype=np.float32)
                    np.savez_compressed(path, iq=iq)
                    saved.append({**info, 'relative_path': relative, 'file_sha256': sha256_file(path),
                                  'sample_length': 256, 'component_count': 2, 'dtype': 'float32'})
        with (cls.prepared / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(saved[0]))
            writer.writeheader()
            writer.writerows(saved)
        write_json(cls.prepared / 'skipped_groups.json', {'representation': 'equalized', 'limited_output_for_code_check': False,
                                                        'skipped_group_count': len(skipped), 'groups': skipped})
        write_json(cls.prepared / 'dataset_summary.json', {'representation': 'equalized', 'limited_output_for_code_check': False,
                   'source_sha256': sha256_file(cls.source), 'group_count': len(saved), 'skipped_empty_group_count': len(skipped),
                   'source_group_count': len(saved) + len(skipped), 'total_signal_count': sum(r['signal_count'] for r in saved)})
        cls.rows, cls.report = build_profile(cls.prepared)
        cls.splits = cls.folder / 'splits'
        cls.index = build_splits(cls.prepared, cls.config_path, cls.splits)
        cls.plan = json.loads((cls.splits / 'fold_00.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_empty_is_distinct_from_nonfinite(self):
        quality = summarize_array(np.empty((0, 256, 2)))
        self.assertTrue(quality['empty'])
        self.assertEqual(quality['non_finite_count'], 0)
        self.assertIsNone(quality['finite_fraction'])
        self.assertFalse(summarize_array(np.empty((0, 255, 2)))['shape_valid'])
        bad = np.ones((1, 256, 2))
        bad[0, 0, 0] = np.nan
        self.assertEqual(summarize_array(bad)['non_finite_count'], 1)

    def test_prepare_skips_only_empty_and_preserves_signal_values(self):
        import contextlib
        import io
        first = np.arange(1024, dtype=np.float64).reshape(2, 256, 2)
        arrays = [first, np.empty((0, 256, 2)), np.ones((1, 256, 2))]
        source = {'tx_list': ['tx'], 'rx_list': ['rx'], 'capture_date_list': ['d0', 'd1', 'd2'],
                  'equalized_list': [0, 1], 'max_sig': 200, 'data': [[[[a, a] for a in arrays]]]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'source.pkl'
            raw.write_bytes(b'mocked input')
            output = root / 'prepared'
            args = ['prepare', '--input', str(raw), '--output-root', str(output), '--representation', 'equalized']
            with patch.object(prepare, 'load_manyrx', return_value=source), patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                prepare.main()
            summary = json.loads((output / 'dataset_summary.json').read_text())
            self.assertEqual((summary['group_count'], summary['skipped_empty_group_count'], summary['total_signal_count']), (2, 1, 3))
            skipped = json.loads((output / 'skipped_groups.json').read_text())
            self.assertEqual(skipped['groups'][0]['capture_date'], 'd1')
            with np.load(output / 'signals/equalized/tx_000__rx_000__day_000.npz') as payload:
                np.testing.assert_array_equal(payload['iq'], first.astype(np.float32))

    def test_prepare_real_nonfinite_and_cast_overflow_still_stop(self):
        for value in (np.nan, np.inf, 1e40):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                raw = root / 'source.pkl'
                raw.write_bytes(b'mocked input')
                array = np.full((1, 256, 2), value)
                source = {'tx_list': ['tx'], 'rx_list': ['rx'], 'capture_date_list': ['d0'],
                          'equalized_list': [0, 1], 'max_sig': 200, 'data': [[[[array, array]]]]}
                args = ['prepare', '--input', str(raw), '--output-root', str(root / 'prepared'), '--representation', 'equalized']
                with patch.object(prepare, 'load_manyrx', return_value=source), patch.object(sys, 'argv', args):
                    with self.assertRaisesRegex(RuntimeError, 'NaN または無限大'):
                        prepare.main()

    def test_coverage_includes_empty_and_short_groups(self):
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(sum(r['signal_count'] == 0 for r in self.rows), 1)
        self.assertEqual(len(self.report['nonzero_below_200_groups']), 1)
        self.assertEqual(self.report['all_conditions_at_least_200_rx_ids'], ['rx_0', 'rx_1'])

    def test_profile_rejects_duplicate_missing_limited_and_inconsistent_metadata(self):
        for mutation in ('duplicate', 'missing', 'limited', 'count'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for filename in ('manifest.csv', 'skipped_groups.json', 'dataset_summary.json'):
                    shutil.copyfile(self.prepared / filename, root / filename)
                if mutation in ('duplicate', 'missing'):
                    with (root / 'manifest.csv').open(newline='', encoding='utf-8') as handle:
                        rows = list(csv.DictReader(handle))
                    fields = list(rows[0])
                    rows = rows + [rows[0]] if mutation == 'duplicate' else rows[1:]
                    with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    summary = json.loads((root / 'dataset_summary.json').read_text())
                    if mutation == 'limited':
                        summary['limited_output_for_code_check'] = True
                    else:
                        summary['total_signal_count'] += 1
                    write_json(root / 'dataset_summary.json', summary)
                with self.assertRaises(ValueError):
                    build_profile(root)

    def test_folds_rotate_unknown_transmitters(self):
        known, development, test = {}, {}, {}
        for entry in self.index['folds']:
            for role, counter in [('known', known), ('validation_unknown', development), ('test_unknown', test)]:
                for tx in entry['tx_roles'][role]:
                    counter[tx] = counter.get(tx, 0) + 1
        self.assertEqual(set(known.values()), {3})
        self.assertEqual(set(development.values()), {1})
        self.assertEqual(set(test.values()), {1})

    def test_primary_counts_and_leakage(self):
        result = validate_fold(self.plan, self.rows, self.report)
        self.assertEqual(result['signal_counts']['train'], 6 * 2 * 120)
        self.assertEqual(result['signal_counts']['validation_known'], 6 * 2 * 40)
        self.assertEqual(result['signal_counts']['validation_unknown'], 2 * 2 * 40)
        def identities(view):
            return {(g['relative_path'], i) for g in self.plan['views'][view] for i in g['signal_indices']}
        train = identities('train')
        valid = identities('validation_known') | identities('validation_unknown')
        tests = set().union(*(identities(name) for name in self.plan['views'] if 'test_' in name))
        self.assertFalse(train & valid or train & tests or valid & tests)
        self.assertTrue(identities('test_2021_03_01') <= identities('supplemental_test_2021_03_01'))

    def test_empty_and_short_groups_remain_in_supplemental_views(self):
        found_empty = found_short = False
        for fold in range(5):
            plan = build_fold(self.rows, self.report, self.config, fold)
            for name, groups in plan['views'].items():
                if not name.startswith('supplemental'):
                    continue
                for group in groups:
                    if group['signal_count'] == 0:
                        found_empty = True
                        self.assertEqual(group['signal_indices'], [])
                    if group['signal_count'] == 5:
                        found_short = True
                        self.assertEqual(len(group['signal_indices']), 5)
        self.assertTrue(found_empty and found_short)

    def test_unused_counts_cover_all_source_signals(self):
        allocations = allocation_rows(self.plan, self.rows)
        for row in allocations:
            self.assertEqual(row['signal_count'], row['selected_unique_signal_count'] + row['unused_signal_count'])
        self.assertEqual(len(allocations), 120)

    def test_paired_supplemental_conditions_and_counts_match_across_days(self):
        for fold in range(5):
            plan = build_fold(self.rows, self.report, self.config, fold)
            pairs_by_day = []
            for date in self.config['test_dates']:
                groups = plan['views'][f'paired_supplemental_test_{date}']
                pairs_by_day.append({(g['tx_id'], g['rx_id']): len(g['signal_indices']) for g in groups})
                self.assertTrue(all(g['signal_indices'] for g in groups))
            self.assertTrue(all(pairs == pairs_by_day[0] for pairs in pairs_by_day))

    def test_rebuilding_is_byte_identical(self):
        second = self.folder / 'splits_rebuilt'
        index = build_splits(self.prepared, self.config_path, second)
        self.assertEqual(self.index, index)
        self.assertEqual(sha256_file(self.splits / 'fold_00.json'), sha256_file(second / 'fold_00.json'))
        self.assertEqual(ranked(range(100), 42, 'test'), ranked(reversed(range(100)), 42, 'test'))

    def test_role_label_index_and_missing_group_tampering_are_rejected(self):
        mutations = ('unknown_in_train', 'wrong_label', 'duplicate_index', 'out_of_range', 'missing_group', 'provenance')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                plan = deepcopy(self.plan)
                group = plan['views']['train'][0]
                if mutation == 'unknown_in_train':
                    group['tx_id'] = plan['tx_roles']['test_unknown'][0]
                elif mutation == 'wrong_label':
                    group['label'] = -1
                elif mutation == 'duplicate_index':
                    group['signal_indices'].append(group['signal_indices'][0])
                elif mutation == 'out_of_range':
                    group['signal_indices'][-1] = 200
                elif mutation == 'missing_group':
                    plan['views']['train'].pop()
                else:
                    plan['provenance']['manifest_sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    validate_fold(plan, self.rows, self.report)

    def test_primary_cohort_with_missing_signals_is_rejected(self):
        config = deepcopy(self.config)
        config['primary_rx_ids'].append('rx_2')
        with self.assertRaises(ValueError):
            validate_config(config, self.report)

    def test_zero_and_nonfinite_normalization_stop(self):
        for value in (np.zeros((256, 2)), np.full((256, 2), np.nan), np.full((256, 2), np.inf)):
            with self.assertRaises(ValueError):
                normalize_iq(value, 'sample_rms')
        with self.assertRaises(ValueError):
            normalize_iq(np.ones((2, 256)), 'sample_rms')

    def test_rms_preserves_phase_and_none_preserves_values(self):
        raw = np.tile(np.array([3, 4], dtype=np.float32), (256, 1))
        normalized = normalize_iq(raw, 'sample_rms')
        np.testing.assert_allclose(normalized, np.tile([0.6, 0.8], (256, 1)), atol=1e-7)
        np.testing.assert_array_equal(normalize_iq(raw, 'none'), raw)
        self.assertEqual(normalized.dtype, np.float32)

    def test_loader_batch_and_unknown_label(self):
        known = WisigDataset(self.prepared, self.splits / 'fold_00.json', 'train', cache_groups=2)
        x, y, metadata = next(known.iter_batches(8, shuffle_seed=7))
        self.assertEqual(x.shape, (8, 256, 2))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertTrue(np.all(y >= 0))
        self.assertEqual(len(metadata), 8)
        unknown = WisigDataset(self.prepared, self.splits / 'fold_00.json', 'validation_unknown')
        self.assertEqual(unknown[0][1], -1)
        with self.assertRaises(ValueError):
            next(known.iter_batches(0))

    def test_cache_cannot_be_modified_through_returned_sample(self):
        data = WisigDataset(self.prepared, self.splits / 'fold_00.json', 'train')
        original = data[0][0]
        data[0][0][:] = 99
        np.testing.assert_array_equal(data[0][0], original)

    def test_path_escape_is_rejected(self):
        for path in ('../outside.npz', 'C:/outside.npz', '/outside.npz', 'signals/not_npz.txt'):
            with self.assertRaises(ValueError):
                safe_signal_path(self.prepared, path)

    def test_waveform_hash_mismatch_is_rejected(self):
        data = WisigDataset(self.prepared, self.splits / 'fold_00.json', 'train')
        path = self.prepared / data.groups[0]['relative_path']
        original = path.read_bytes()
        try:
            path.write_bytes(b'corrupted npz')
            with self.assertRaisesRegex(ValueError, 'ハッシュ不一致'):
                data[0]
        finally:
            path.write_bytes(original)

    def test_full_verification(self):
        report = verify_data(self.prepared, self.splits, self.source)
        self.assertTrue(report['verification_passed'])
        self.assertTrue(report['source_file_hash_checked'])
        self.assertFalse(report['experiment_executed'])
        self.assertEqual(report['fold_count'], 5)
        self.assertEqual(report['loader_smoke_check_count'], 20)


if __name__ == '__main__':
    unittest.main()
