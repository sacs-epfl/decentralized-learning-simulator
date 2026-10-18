"""Run on sacs006 after both accelerated short validations complete."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path('/mnt/data/mdevos/conflux')
NEW = ROOT / 'results/tiny128-gpu-training'
SOURCE = ROOT / 'gpu-backend-code'
EXPECTED_SOURCE = '99e4f916043d7a761735f730c0473f7728c49b93b6087a86956dff29eafbcdea'
digest = hashlib.sha256()
for path in sorted([*(SOURCE/'dasklearn').rglob('*.py'), *(SOURCE/'scripts').glob('*.py')]):
    digest.update(str(path.relative_to(SOURCE)).encode())
    digest.update(path.read_bytes())
assert digest.hexdigest() == EXPECTED_SOURCE


def frame(path):
    return pd.read_csv(path).sort_values(['time', 'peer']).reset_index(drop=True)


report = {'passed': True, 'execution_commit': '03da38a',
          'source_sha256': EXPECTED_SOURCE, 'variants': {},
          'tests': {'linux_cuda': '62 passed', 'macos_cpu': '56 passed, 6 CUDA checks skipped'},
          'timing_note': 'Observed wall times on separate GPUs sharing the server; not controlled timing trials. Training microbenchmark uses three repetitions on one GPU.'}
for variant in ['shatter-r2', 'conflux-v2']:
    new = NEW / variant
    assert (new/'COMPLETE').exists(), variant
    manifest = json.loads((new/'manifest.json').read_text())
    assert manifest['source_sha256'] == EXPECTED_SOURCE
    assert manifest['gpu_residency'] and manifest['training_batching']
    assert manifest['training_batch_size'] == 2
    snapshots = frame(new/'runs'/variant/'accuracies.csv')
    final = frame(new/'final.csv')
    assert len(snapshots) == 768 and len(final) == 128
    assert snapshots.groupby('time').peer.nunique().eq(128).all()
    assert sorted(snapshots.time.unique()) == [300, 600, 900, 1200, 1500, 1800]
    assert final.peer.nunique() == 128
    assert np.isfinite(snapshots[['accuracy', 'loss']]).all().all()
    assert np.isfinite(final[['accuracy', 'loss']]).all().all()
    with (new/'runs'/variant/'workflow_dag').open('rb') as handle:
        dag = pickle.load(handle)
    summary = pd.read_csv(new/'summary.csv').iloc[0]
    comparisons = []
    for previous in ['tiny128', 'tiny128-optimized'] + (['tiny128-memoized'] if variant == 'conflux-v2' else []):
        old = ROOT/'results'/previous/variant
        pd.testing.assert_frame_equal(snapshots, frame(old/'runs'/variant/'accuracies.csv'), check_exact=True)
        pd.testing.assert_frame_equal(final, frame(old/'final.csv'), check_exact=True)
        prior_manifest = json.loads((old/'manifest.json').read_text())
        excluded = {'output', 'source_sha256', 'cuda_visible_devices',
                    'gpu_residency', 'training_batching', 'training_batch_size', 'gpu_state_cache_mb'}
        for key in prior_manifest.keys() - excluded:
            assert manifest[key] == prior_manifest[key], (variant, key)
        with (old/'runs'/variant/'workflow_dag').open('rb') as handle:
            assert dag == pickle.load(handle)
        prior = pd.read_csv(old/'summary.csv').iloc[0]
        for key in summary.index.drop('actual_execution_s'):
            assert summary[key] == prior[key], (variant, key)
        comparisons.append({'reference': previous, 'per_node_curve_rows_exact': len(snapshots),
                            'full_validation_rows_exact': len(final), 'complete_dag_exact': True,
                            'manifest_shared_fields_exact': True, 'summary_except_wall_time_exact': True,
                            'previous_execution_s': float(prior.actual_execution_s),
                            'speedup_vs_previous': float(prior.actual_execution_s / summary.actual_execution_s)})
    status = None
    log = ROOT/'logs'/f'tiny128-gpu-training-{variant}.log'
    for line in log.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'completed' in entry:
            status = entry
    assert status['completed'] == status['total'] == len(dag)
    backend = json.loads((new/'training-executor.json').read_text())
    assert backend['prefetch_peak_models'] <= 2
    assert backend['input_cache_bytes'] == 0
    report['variants'][variant] = {'execution_s': float(summary.actual_execution_s),
        'comparisons': comparisons, 'final_status': status, 'training_executor': backend}

(NEW/'equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
