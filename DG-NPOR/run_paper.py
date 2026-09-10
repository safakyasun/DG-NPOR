#!/usr/bin/env python3
"""Frozen DG-NPOR + local ParticleNet + same-jet GN2/DL1d paper results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from frozen_inputs import source_identity, sha256, save_json, validate_splits, read_jets, metadata_labels

ROOT = Path(__file__).resolve().parent
DG_MODEL = 'trained_self_configuring_dg_npor_v9.joblib'
DG_SCORES = 'self_configuring_locked_test_predictions.csv.gz'
PN_SCORES = 'particle_net_locked_test_predictions.csv.gz'
PN_WP = 'particle_net_wp_validation_predictions.csv.gz'
PN_MANIFEST = 'particle_net_manifest.json'
KEYS = ['source_row', 'event_number', 'y_true_light0_b1']


def say(message):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def activate(root=ROOT):
    for path in reversed([root / 'engines/dg_npor', root / 'engines/dg_npor/src', root / 'engines/dg_npor/examples', root / 'engines/paper/src']):
        sys.path.insert(0, str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def search_roots():
    roots = [ROOT, Path.home() / 'paper']
    for pattern in ('ATLAS_JETSET*', 'DG_NPOR*'):
        roots.extend(sorted((Path.home() / 'Downloads').glob(pattern)))
    return list(dict.fromkeys(p.resolve() for p in roots if p.is_dir()))


def find_files(filename):
    found = set()
    for root in search_roots():
        for folder, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('engines', '__pycache__', 'env', 'venv', 'node_modules'))
            if len(Path(folder).relative_to(root).parts) >= 5:
                dirs[:] = []
            if filename in files:
                found.add((Path(folder) / filename).resolve())
    return sorted(found)


def resolve_input(explicit, known, filename, description):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path = path / filename if filename and path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError('%s bulunamadı: %s' % (description, path))
        return path
    for value in known:
        path = Path(value).expanduser()
        path = path / filename if filename else path
        if path.is_file():
            return path.resolve()
    candidates = find_files(filename or 'mc-flavtag-ttbar-small.h5')
    if filename == DG_MODEL:
        candidates = [p for p in candidates if '500k' in str(p.parent).lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and len({sha256(p) for p in candidates}) == 1:
        return candidates[0]
    details = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError('%s otomatik olarak belirlenemedi. --model-dir veya --h5-file ile mevcut yolu belirt.\n%s' % (description, details))


def expected_frame(source_rows, events, indices, labels):
    import pandas as pd
    return pd.DataFrame({'source_row': source_rows[indices], 'event_number': events[indices], 'y_true_light0_b1': labels})


def align_predictions(frame, expected, column):
    import numpy as np
    if not set(KEYS + [column]).issubset(frame.columns):
        raise ValueError('Missing prediction columns: ' + column)
    if len(frame) != len(expected) or frame.source_row.duplicated().any():
        raise ValueError('Prediction count or unique source-row identity differs from frozen evaluation.')
    aligned = expected.merge(frame[KEYS + [column]], on=KEYS, how='left', sort=False, validate='one_to_one')
    values = aligned[column].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Prediction keys/labels or probabilities do not match the frozen evaluation.')
    return aligned


def validate_particle(directory, test_expected, wp_expected, roles, bundle):
    import numpy as np
    import pandas as pd
    directory = Path(directory).resolve()
    manifest = read_json(directory / PN_MANIFEST)
    if manifest.get('architecture') != 'full' or int(manifest.get('model', {}).get('paper_dense_units', 0)) != 256:
        raise ValueError('Full ParticleNet with a 256-unit final hidden layer is required.')
    if manifest.get('selection', {}).get('locked_test_used_for_selection') is not False:
        raise ValueError('ParticleNet validation-only checkpoint selection is not documented.')
    if manifest.get('truth_gn2_dl1_used_as_input') is not False:
        raise ValueError('ParticleNet input-field provenance is incomplete.')
    inputs = manifest.get('inputs', {})
    legacy = 'max_tracks' not in inputs
    maximum = int(inputs.get('max_tracks', 20))
    if maximum != int(bundle['max_tracks']):
        raise ValueError('ParticleNet and DG-NPOR use different track limits.')
    model_info = manifest.get('model', {})
    if int(model_info.get('jetset_track_feature_count', 0)) != 19 or int(model_info.get('jet_context_feature_count', 0)) != 4:
        raise ValueError('ParticleNet input dimensions do not match the local JetSet adapter.')
    counts = manifest.get('frozen_roles', {})
    expected_counts = {'train': sum(len(roles[k]) for k in ('base', 'operator_validation', 'derivative_gate')),
                       'validation': len(wp_expected), 'locked_test': len(test_expected)}
    if any(int(counts.get(k, -1)) != v for k, v in expected_counts.items()):
        raise ValueError('ParticleNet frozen role counts differ.')
    training = manifest.get('training', {})
    completed = int(training.get('completed_epochs', 0))
    requested = int(training.get('requested_epochs', 0))
    if completed < 1 or requested < 1 or completed > requested:
        raise ValueError('ParticleNet completed training could not be verified.')
    if training.get('preset') == 'paper' and completed != requested:
        raise ValueError('ParticleNet paper training is incomplete.')
    source_split_status = 'legacy metadata, row keys and role counts; original split path unavailable'
    source_dir = Path(manifest.get('source_selfconfig_output', '')).expanduser()
    if (source_dir / 'frozen_split_indices.npz').is_file():
        with np.load(source_dir / 'frozen_split_indices.npz', allow_pickle=False) as split:
            if not np.array_equal(split['sampled_source_rows'], bundle['sampled_source_rows']) or not np.array_equal(split['sampled_event_numbers'], bundle['sampled_event_numbers']):
                raise ValueError('ParticleNet used a different frozen source sample.')
            for role, indices in roles.items():
                if not np.array_equal(split[role], indices):
                    raise ValueError('ParticleNet source split differs in role: ' + role)
        source_split_status = 'exact source split arrays verified'
    if not (directory / PN_WP).is_file():
        raise ValueError('ParticleNet WP-validation predictions are required to verify the development population.')
    align_predictions(pd.read_csv(directory / PN_WP), wp_expected, 'particle_net_probability_b')
    if not (directory / PN_SCORES).is_file():
        raise ValueError('ParticleNet completed test prediction file is missing.')
    aligned = align_predictions(pd.read_csv(directory / PN_SCORES), test_expected, 'particle_net_probability_b')
    return manifest, aligned, {'directory': str(directory), 'source_split_verification': source_split_status,
        'legacy_track_limit_assumed_20': legacy, 'manifest_sha256': sha256(directory / PN_MANIFEST),
        'prediction_sha256': sha256(directory / PN_SCORES), 'validation_prediction_sha256': sha256(directory / PN_WP),
        'implementation': 'Existing local JetSet adaptation: max EdgeConv aggregation, self-neighbors and ordinary batch normalization; unchanged in this release.'}


def particle_candidates(explicit, config, output):
    if explicit:
        return [Path(explicit).expanduser().resolve()]
    candidates = [output / 'particlenet'] + [Path(p).expanduser() for p in config['particle_net_dirs']]
    candidates += [p.parent for p in find_files(PN_MANIFEST)]
    unique = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
    # Declared completed paper schedule takes priority; metrics are never used.
    def priority(p):
        try:
            m = read_json(p / PN_MANIFEST)
            return 0 if m.get('training', {}).get('preset') == 'paper' else 1
        except (OSError, ValueError):
            return 2
    return sorted(unique, key=priority)


def get_particle(args, config, output, model_dir, h5, test_expected, wp_expected, roles, bundle):
    rejections = []
    for path in particle_candidates(args.particle_net_dir, config, output):
        try:
            manifest, aligned, provenance = validate_particle(path, test_expected, wp_expected, roles, bundle)
            say('Tamamlanmış ParticleNet kullanılıyor: ' + str(path))
            return manifest, aligned, provenance, rejections
        except (OSError, ValueError, KeyError) as exc:
            rejections.append({'directory': str(path), 'reason': str(exc)})
            if args.particle_net_dir:
                raise RuntimeError('Belirtilen ParticleNet sonucu uygun değil: ' + str(exc)) from exc
    save_json(output / 'PARTICLENET_DISCOVERY.json', {'rejected_candidates': rejections})
    if not args.train_particle_net_if_missing:
        raise FileNotFoundError('Tamamlanmış ve aynı jetlerde değerlendirilmiş ParticleNet bulunamadı. --particle-net-dir ile klasörünü belirt veya --train-particle-net-if-missing kullan.')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('ParticleNet eğitimi için PyTorch gerekli. Komut: "%s" -m pip install "torch>=2.5,<3"' % sys.executable) from exc
    from jetset_litcomp.training import train_particle_net
    destination = output / 'particlenet'
    if destination.exists() and any(destination.iterdir()):
        backup = output / ('particlenet_previous_attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        destination.rename(backup)
        say('Önceki eksik deneme korundu: ' + str(backup))
    say('Uygun bitmiş ParticleNet yok. Full mimari, paper programı (20 epoch) eğitiliyor.')
    say('Eğitim development rollerini kullanır; DG-NPOR modeli değişmez. Çıktı: ' + str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'PAPER_RUN_CONTEXT.json', {'model_sha256': sha256(model_dir / DG_MODEL),
        'split_sha256': sha256(model_dir / 'frozen_split_indices.npz'), 'paper_code_sha256': sha256(ROOT / 'engines/paper/src/jetset_litcomp/particle_net.py'),
        'preset': 'paper', 'architecture': 'full', 'seed': 42, 'max_tracks': int(bundle['max_tracks'])})
    train_particle_net(model_dir, h5, destination, architecture='full', preset='paper', device=args.device,
        seed=42, batch_size=args.particle_batch_size, max_tracks=int(bundle['max_tracks']))
    manifest, aligned, provenance = validate_particle(destination, test_expected, wp_expected, roles, bundle)
    return manifest, aligned, provenance, rejections


def dg_states_and_scores(bundle, model_dir, h5, expected, test_indices, output, read_batch):
    import numpy as np
    import pandas as pd
    from por_hep.structured_tracks import H5StructuredTrackSource
    cached = output / 'FROZEN_DG_TEST.npz'
    receipt_path = output / 'DG_SCORE_CACHE.json'
    if cached.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get('file_sha256') != sha256(cached) or receipt.get('model_sha256') != sha256(model_dir / DG_MODEL):
            raise ValueError('Frozen DG test cache identity mismatch.')
        with np.load(cached, allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
        for key in KEYS:
            if not np.array_equal(values[key], expected[key]):
                raise ValueError('Frozen DG cache evaluation keys differ.')
        say('Kayıtlı sabit DG-NPOR test temsili yeniden kullanılıyor.')
        return values['normalized_state'], values['probability_b'], receipt
    stored_file = model_dir / DG_SCORES
    status = model_dir / 'test_scoring_status.json'
    if status.is_file() and read_json(status).get('scored') and not stored_file.is_file():
        raise FileNotFoundError('Önceki kayıt testin skorlandığını söylüyor fakat tahmin CSV dosyası yok. Özgün %s dosyasını model klasörüne geri koy.' % DG_SCORES)
    stored = None
    if stored_file.is_file():
        stored = align_predictions(pd.read_csv(stored_file), expected, 'self_configuring_por_probability_b')
    encoder, adapter, model = (bundle[k] for k in ('structured_track_encoder', 'input_feature_adapter', 'neural_por_model'))
    encodings = []
    say('Sabit encoder ile test jetleri okunuyor; seçilmiş orbitaller dışa aktarılacak.')
    with H5StructuredTrackSource(h5, bundle['sampled_source_rows'], max_tracks=int(bundle['max_tracks'])) as source:
        for start in range(0, len(test_indices), read_batch):
            raw = source.read(test_indices[start:start + read_batch])
            geometry, _ = encoder.transform_arrays(*raw)
            encodings.append(geometry)
            if start == 0 or (start // read_batch + 1) % 20 == 0:
                say('Encoder: %d/%d test jeti' % (min(start + read_batch, len(test_indices)), len(test_indices)))
    e = adapter.transform(np.concatenate(encodings))
    del encodings
    # The original model controls its own Nyström/query batching.
    state = model.transform(e)
    probability = model.measurement_.predict_proba(state)[:, 1]
    if state.shape != (len(expected), int(model.selected_k_)) or not np.isfinite(state).all():
        raise ValueError('Invalid frozen orbital export.')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid frozen DG probabilities.')
    max_difference = None
    if stored is not None:
        old = stored.self_configuring_por_probability_b.to_numpy(float)
        max_difference = float(np.max(np.abs(probability - old)))
        if max_difference > 1e-6:
            raise ValueError('Exported frozen state disagrees with stored test probabilities: %g' % max_difference)
        probability = old  # Publication uses the original stored scores.
    with cached.open('wb') as stream:
        np.savez_compressed(stream, **{key: expected[key].to_numpy() for key in KEYS}, normalized_state=state, probability_b=probability,
                            selected_orbitals_one_based=np.asarray(model.selected_orbital_indices_) + 1)
    receipt = {'model_sha256': sha256(model_dir / DG_MODEL), 'file_sha256': sha256(cached),
        'source': 'original stored predictions verified by state reconstruction' if stored is not None else 'new scoring of existing frozen model',
        'original_prediction_sha256': sha256(stored_file) if stored is not None else None,
        'max_probability_reconstruction_difference': max_difference, 'model_refit': False, 'K_reselected': False}
    save_json(receipt_path, receipt)
    return state, probability, receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--h5-file', type=Path)
    parser.add_argument('--particle-net-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper500k')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--particle-batch-size', type=int, default=256)
    parser.add_argument('--read-batch-size', type=int, default=512)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--train-particle-net-if-missing', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.read_batch_size, args.particle_batch_size) < 1 or args.bootstrap < 20:
        raise ValueError('Positive batches and at least 20 bootstrap repetitions are required.')
    activate()
    import h5py
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from por_hep.atlas_jetset import verify_official_file, inspect_schema, gn2_paper_b_discriminant, dl1d_paper_b_discriminant
    from paper_report import write_paper_results
    config = read_json(ROOT / 'configs/paths.json')
    identity = source_identity(ROOT)
    model_path = resolve_input(args.model_dir, config['model_dirs'], DG_MODEL, 'Eğitilmiş DG-NPOR modeli')
    model_dir = model_path.parent
    h5 = resolve_input(args.h5_file, config['h5_files'], None, 'JetSet H5')
    say('Python: %s | %s' % (platform.python_version(), sys.executable))
    say('Mevcut DG-NPOR: ' + str(model_dir))
    say('H5: ' + str(h5))
    integrity = verify_official_file(h5, tier='small')
    inspect_schema(h5, label_definition='cone')
    bundle = joblib.load(model_path)
    if bundle.get('source_tree_sha256') != identity['source_tree_sha256']:
        raise ValueError('The existing DG-NPOR model source hash differs from the bundled unchanged source.')
    for name, expected in [('analysis_protocol', 'paper-ttbar'), ('label_definition', 'cone'), ('dataset_tier', 'small'), ('method_resolved_before_test', True)]:
        if bundle.get(name) != expected:
            raise ValueError('Unsupported model protocol: ' + name)
    if bundle.get('official_file_size') != integrity['size_bytes'] or bundle.get('official_file_adler32') != integrity['adler32']:
        raise ValueError('H5 identity differs from the frozen model.')
    with np.load(model_dir / 'frozen_split_indices.npz', allow_pickle=False) as split:
        splits = {key: split[key] for key in split.files}
    source_rows, events, roles, measurement_train = validate_splits(bundle, splits)
    model = bundle['neural_por_model']
    if not np.array_equal(model.label_encoder_.classes_, [0, 1]):
        raise ValueError('Unexpected DG class encoding.')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = {'dg_model_sha256': sha256(model_path), 'split_sha256': sha256(model_dir / 'frozen_split_indices.npz'),
        'source_tree_sha256': identity['source_tree_sha256'], 'official_h5_size': integrity['size_bytes'], 'official_h5_adler32': integrity['adler32'],
        'frozen_test_rows': len(roles['independent_test']), 'K_DG': int(model.selected_k_), 'selected_orbitals': np.asarray(model.selected_orbital_indices_).tolist()}
    if (output / 'RUN_CONTEXT.json').is_file() and read_json(output / 'RUN_CONTEXT.json') != context:
        raise RuntimeError('This output belongs to a different frozen experiment. Choose a new --output.')
    save_json(output / 'RUN_CONTEXT.json', context)
    test, wp = roles['independent_test'], roles['wp_validation']
    test_jets = read_jets(h5, source_rows[test])
    wp_jets = read_jets(h5, source_rows[wp])
    y_test = metadata_labels(test_jets, events[test], bundle['label_field'])
    y_wp = metadata_labels(wp_jets, events[wp], bundle['label_field'])
    test_expected = expected_frame(source_rows, events, test, y_test)
    wp_expected = expected_frame(source_rows, events, wp, y_wp)
    say('Aynı nihai test: %d jet (%d b, %d hafif), %d event. K_PF=%d, K_DG=%d.' %
        (len(test), (y_test == 1).sum(), (y_test == 0).sum(), len(np.unique(events[test])), model.candidate_bank_k_, model.selected_k_))
    # Resolve the baseline by declared configuration/alignment, without ranking its test scores.
    pn_manifest, pn_scores, pn_provenance, rejected = get_particle(args, config, output, model_dir, h5, test_expected, wp_expected, roles, bundle)
    pn_identity = {key: pn_provenance[key] for key in ('manifest_sha256', 'prediction_sha256', 'validation_prediction_sha256')}
    pn_identity_path = output / 'PN_INPUT_IDENTITY.json'
    if pn_identity_path.is_file() and read_json(pn_identity_path) != pn_identity:
        raise RuntimeError('This output is bound to another ParticleNet run. Choose a new --output to preserve the previous comparison.')
    save_json(pn_identity_path, pn_identity)
    state, dg_probability, dg_receipt = dg_states_and_scores(bundle, model_dir, h5, test_expected, test, output, args.read_batch_size)
    scores = {'DG-NPOR': dg_probability, 'ParticleNet (JetSet)': pn_scores.particle_net_probability_b.to_numpy(float)}
    for name, fields, function in [('GN2v01', ['GN2v01_pb', 'GN2v01_pc', 'GN2v01_ptau', 'GN2v01_pu'], gn2_paper_b_discriminant),
                                   ('DL1dv01', ['DL1dv01_pb', 'DL1dv01_pc', 'DL1dv01_pu'], dl1d_paper_b_discriminant)]:
        arrays = [np.asarray(test_jets[field], dtype=float) for field in fields]
        if not all(np.isfinite(a).all() and np.all((a >= 0) & (a <= 1)) for a in arrays):
            raise ValueError('Invalid stored ' + name + ' probabilities. No rows will be silently removed.')
        scores[name] = function(*arrays)
    predictions = test_expected.assign(pt_GeV=np.asarray(test_jets['pt_btagJes'], dtype=float) / 1000,
        eta=np.asarray(test_jets['eta_btagJes'], dtype=float), **scores)
    predictions.to_csv(output / 'COMMON_TEST_PREDICTIONS.csv.gz', index=False, compression='gzip')
    geometrical = model.neural_geometry_
    inner_overlap = len(np.intersect1d(events[roles['base'][geometrical.fit_indices_]], events[roles['base'][geometrical.validation_indices_]]))
    report_meta = {'dataset': 'CERN JetSet simulated ttbar, small file', 'evaluation_role': 'frozen independent_test, eventNumber modulo 10 equals 9',
        'sample_jets': len(source_rows), 'test_jets': len(test), 'test_events': len(np.unique(events[test])), 'test_b': int(y_test.sum()), 'test_light': int((y_test == 0).sum()),
        'K_PF': int(model.candidate_bank_k_), 'K_DG': int(model.selected_k_), 'selected_orbitals_one_based': np.asarray(model.selected_orbital_indices_) + 1,
        'encoder_dimension': int(bundle['input_feature_adapter'].dimension), 'selected_architecture': bundle['selected_architecture'],
        'max_tracks': int(bundle['max_tracks']), 'particle_net': pn_manifest, 'particle_net_provenance': pn_provenance,
        'roles': {k: {'jets': len(v), 'events': len(np.unique(events[v]))} for k, v in roles.items()},
        'encoder_train_jets': len(splits['encoder_train']), 'metric_train_jets': len(geometrical.fit_indices_),
        'metric_validation_jets': len(geometrical.validation_indices_), 'metric_inner_overlap_events': inner_overlap,
        'landmark_jets': int(len(model.core_.base.X)), 'PSD_train_jets': len(measurement_train),
        'dg_score_receipt': dg_receipt, 'source_identity': identity, 'h5_identity': integrity,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'h5py': h5py.__version__},
        'future_work_implemented': False, 'new_PSD_controls_in_paper': False,
        'uncertainty_scope': 'conditional on fixed fitted models; not variation across training runs',
        'test_history': 'Stored scores reused when present. No claim that this test has never been examined in earlier work.'}
    save_json(output / 'COMPARISON_METADATA.json', report_meta)
    write_paper_results(predictions, state, report_meta, output, args.bootstrap)
    if source_identity(ROOT) != identity:
        raise RuntimeError('Original DG source changed during the run.')
    if sha256(model_path) != context['dg_model_sha256']:
        raise RuntimeError('The frozen model changed during the run.')
    archive = output / 'PAPER_RESULTS.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ('paper', 'statistics', 'figures', 'overleaf'):
            for path in sorted((output / folder).rglob('*')):
                if path.is_file() and path.suffix not in ('.log', '.aux', '.out'):
                    z.write(path, str(path.relative_to(output)))
        for filename in ('COMPARISON_METADATA.json', 'RUN_CONTEXT.json', 'PN_INPUT_IDENTITY.json', 'COMMON_TEST_PREDICTIONS.csv.gz', 'DG_SCORE_CACHE.json', 'OVERLEAF_RESULTS.zip'):
            z.write(output / filename, filename)
    say('Makale dosyaları hazır: ' + str(output / 'paper'))
    say('Paylaşılacak dosya: ' + str(archive))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('\nPAPER ÇIKTILARI TAMAMLANMADI: %s' % error, file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
