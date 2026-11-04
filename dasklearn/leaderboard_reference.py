"""Attach a separately trained pooled-data reference without altering rankings."""
import json
import math
from pathlib import Path


def reference_spec(catalog):
    if catalog is None:
        return None
    path = Path(catalog).resolve()
    spec = json.loads(path.read_text()).get('reference')
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {'output'}
            or not isinstance(spec['output'], str) or not spec['output'].strip()):
        raise ValueError('reference requires a nonempty output directory')
    return dict(output=(path.parent / Path(spec['output']).expanduser()).resolve())


def centralized_reference(target, spec):
    """Never mix validation trials or incompatible datasets with held-out lines."""
    if (spec is None or target.get('evaluation_split') != 'test'
            or target.get('evaluation_model') != 'global'
            or target.get('scenario', {}).get('peers', 0) <= 1):
        return None
    output = spec['output']
    result = dict(name='Centralized', status='pending', accuracy=None, loss=None,
                  source_output=str(output), time_basis='optimizer_steps')
    try:
        reference = json.loads((output / 'reference.json').read_text())
        settings = reference['settings']
        shared = ('dataset', 'model', 'dataset_base_path', 'dataset_name', 'dataset_revision',
                  'dataset_config', 'dataset_data_files', 'train_split', 'test_split', 'validation_split',
                  'validation_set_fraction', 'feature_column', 'label_column')
        training = ('batch_size', 'local_steps', 'momentum', 'weight_decay')
        if (reference['kind'] != 'centralized-reference' or reference['time_basis'] != 'optimizer_steps'
                or settings['algorithm'] != 'centralized' or settings['participants'] != 1
                or settings['partitioner'] != 'iid' or settings['evaluation_split'] != 'test'
                or settings['evaluation_examples'] != 0
                or reference['seeds'] != sorted({run['seed'] for run in target['runs']})
                or any(any(run['options'].get(key) != settings.get(key) for key in shared)
                       or any(run['options'].get(key) != settings['learning'][key] for key in training)
                       or run['options']['evaluation_examples'] != 0 for run in target['runs'])):
            return result | dict(status='incompatible', error='Reference data, model, optimizer, seeds or evaluation differ')
        result.update({key: reference.get(key) for key in
            ('status', 'phase', 'steps', 'learning_rate', 'seeds', 'checkpoint_rule', 'selection_rule',
             'accuracy', 'loss', 'error', 'provenance', 'updated_at')})
        result['runs'] = [{key: run.get(key) for key in ('name', 'seed', 'status', 'completed_steps', 'total_steps',
            'elapsed_seconds', 'best_validation', 'test', 'training_examples', 'learning_rate', 'error')}
            for run in reference['runs']]
        if reference['status'] != 'completed':
            result.update(accuracy=None, loss=None)
            return result
        if (len(reference['runs']) != len(reference['seeds'])
                or {run['seed'] for run in reference['runs']} != set(reference['seeds'])
                or any(run['status'] != 'completed' or run['completed_steps'] != reference['steps']
                       or run.get('test') is None for run in reference['runs'])
                or any(reference.get(metric) is None or not math.isfinite(reference[metric]['mean'])
                       or reference[metric]['sd'] is not None and not math.isfinite(reference[metric]['sd'])
                       for metric in ('accuracy', 'loss'))
                or not 0 <= reference['accuracy']['mean'] <= 100):
            return result | dict(status='invalid', accuracy=None, loss=None, error='Reference endpoints must be complete and finite')
        return result
    except FileNotFoundError:
        return result
    except (ValueError, KeyError, TypeError) as exc:
        return result | dict(status='invalid', accuracy=None, loss=None, error=f'Cannot read centralized reference: {exc}')
