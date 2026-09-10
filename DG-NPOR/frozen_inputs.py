"""Frozen source, input identity and event split validation. No model fitting."""
import hashlib
import itertools
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def source_identity(workspace):
    engine = workspace / 'engines' / 'dg_npor'
    reference = json.loads((workspace / 'docs/source_provenance.json').read_text())['unchanged_dg_npor_python']
    changed = [name for name, digest in reference.items()
               if not (engine / name).is_file() or sha256(engine / name) != digest]
    if changed:
        raise RuntimeError('Original DG-NPOR source differs: ' + ', '.join(changed))
    paths = list((engine / 'src').rglob('*.py')) + list((engine / 'examples').rglob('*.py')) + list(engine.glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: str(p.relative_to(engine))):
        digest.update(str(path.relative_to(engine)).replace('\\', '/').encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return {'files': len(reference), 'changed': [], 'source_tree_sha256': digest.hexdigest()}

def clean_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def save_json(path, payload):
    Path(path).write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def checked_indices(values, length, name):
    import numpy as np
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
        raise ValueError('%s must be a nonempty integer index array.' % name)
    index = raw.astype(np.int64)
    if index.min() < 0 or index.max() >= length or len(np.unique(index)) != len(index):
        raise ValueError('%s contains duplicate/out-of-range indices.' % name)
    return index

def validate_splits(bundle, splits):
    import numpy as np
    rows = np.asarray(bundle['sampled_source_rows'])
    events = np.asarray(bundle['sampled_event_numbers'])
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows) or rows.min() < 0:
        raise ValueError('Invalid sampled_source_rows.')
    if events.shape != rows.shape or events.dtype.kind not in 'iu':
        raise ValueError('Invalid sampled_event_numbers.')
    for key in ('sampled_source_rows', 'sampled_event_numbers'):
        if not np.array_equal(bundle[key], splits[key]):
            raise ValueError('Bundle/split mismatch: ' + key)
    role_names = ('base', 'operator_validation', 'derivative_gate', 'wp_validation', 'independent_test')
    roles = {name: checked_indices(splits[name], len(rows), name) for name in role_names}
    if not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(len(rows))):
        raise ValueError('The frozen roles do not partition the sampled rows exactly.')
    for left, right in itertools.combinations(role_names, 2):
        if np.intersect1d(events[roles[left]], events[roles[right]]).size:
            raise ValueError('Event overlap: %s / %s' % (left, right))
    expected = {'base': set(range(6)), 'operator_validation': {6, 7}, 'derivative_gate': {8}, 'wp_validation': {8}, 'independent_test': {9}}
    for name in role_names:
        if not set((events[roles[name]] % 10).tolist()).issubset(expected[name]):
            raise ValueError('Unexpected paper-ttbar residue for ' + name)
    dimension = checked_indices(splits['dimension_gate'], len(rows), 'dimension_gate')
    if not np.array_equal(np.sort(dimension), np.sort(np.concatenate([roles['derivative_gate'], roles['wp_validation']]))):
        raise ValueError('Dimension split mismatch.')
    if not np.array_equal(bundle['independent_test_indices'], roles['independent_test']):
        raise ValueError('Final-test mapping mismatch.')
    encoder_sets = []
    for name in ('encoder_train', 'encoder_early_stop', 'architecture_validation'):
        indices = checked_indices(splits[name], len(rows), name)
        if not np.isin(indices, roles['base']).all():
            raise ValueError('Encoder role extends outside base: ' + name)
        encoder_sets.append(events[indices])
    if any(np.intersect1d(a, b).size for a, b in itertools.combinations(encoder_sets, 2)):
        raise ValueError('Encoder roles overlap in events.')
    model = bundle['neural_por_model']
    landmarks = checked_indices(model.selection_training_indices_, len(roles['base']), 'landmarks')
    base_mask = np.ones(len(roles['base']), dtype=bool)
    base_mask[landmarks] = False
    pool = np.concatenate([roles['base'][base_mask], roles['operator_validation'], roles['derivative_gate']])
    train = pool[checked_indices(model.measurement_training_indices_, len(pool), 'measurement_training_indices')]
    if np.intersect1d(events[train], events[roles['wp_validation']]).size:
        raise ValueError('Readout/evaluation events overlap.')
    return rows, events, roles, train

def read_jets(h5_path, source_rows):
    import h5py
    import numpy as np
    from por_hep.atlas_jetset import _read_rows_chunked
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError('Repeated source rows in metadata read.')
    order = np.argsort(source_rows)
    with h5py.File(str(h5_path), 'r') as handle:
        if source_rows.min() < 0 or source_rows.max() >= len(handle['jets']):
            raise ValueError('Source row outside H5.')
        result = _read_rows_chunked(handle['jets'], source_rows[order])
    return result[np.argsort(order)]

def metadata_labels(jets, expected_events, label_field):
    import numpy as np
    if not np.array_equal(jets['eventNumber'], expected_events):
        raise ValueError('H5 events disagree with frozen row mapping.')
    raw = np.asarray(jets[label_field])
    if not set(np.unique(raw)).issubset({0, 5}):
        raise ValueError('Evaluation requires b=5 versus light=0 source labels.')
    for field in ('pt_btagJes', 'eta_btagJes'):
        if not np.isfinite(jets[field]).all():
            raise ValueError('Nonfinite reconstructed context: ' + field)
    return (raw == 5).astype(int)
