"""Small HDF5/software fixtures only, never publishable ATLAS results."""
from pathlib import Path
import importlib.util
import json
import os
import sys

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_paper
run_paper.activate()
from frozen_inputs import source_identity, sha256, validate_splits
from por_hep.atlas_jetset import _required_schema, file_adler32, verify_official_file
from por_hep.structured_tracks import H5StructuredTrackSource, IdentityFeatureMap
from por_hep.self_configuring_geometry import PhysicsConstrainedGatedTrackGeometry
from por_core.estimator_derivative_gated import DerivativeGatedNeuralPOR
from por_core.neural_geometry import NeuralGeometryMap
from por_core.readout import StructuredOrbitalMeasurement
from sklearn.preprocessing import LabelEncoder


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    if importlib.util.find_spec('torch') is None:
        pytest.skip('Optional PyTorch is required for the actual tiny full-ParticleNet training integration.')
    import torch
    torch.set_num_threads(2)
    from jetset_litcomp.training import train_particle_net
    directory = tmp_path_factory.mktemp('SOFTWARE_FIXTURE_NOT_ATLAS')
    model_dir = directory / 'model'
    model_dir.mkdir()
    n = 400
    source_rows = np.arange(n)
    events = np.arange(n)
    base = np.flatnonzero(events % 10 < 6)
    dimension = np.flatnonzero(events % 10 == 8)
    splits = {'sampled_source_rows': source_rows, 'sampled_event_numbers': events, 'base': base,
        'operator_validation': np.flatnonzero(np.isin(events % 10, [6, 7])), 'dimension_gate': dimension,
        'derivative_gate': dimension[::2], 'wp_validation': dimension[1::2], 'independent_test': np.flatnonzero(events % 10 == 9),
        'encoder_train': base[:100], 'encoder_early_stop': base[100:130], 'architecture_validation': base[130:160]}
    np.savez_compressed(model_dir / 'frozen_split_indices.npz', **splits)
    required = _required_schema('HadronConeExclTruthLabelID')
    jets = np.zeros(n, dtype=[(k, 'i8' if k == 'eventNumber' else 'f4') for k in sorted(required['jets'])])
    rng = np.random.RandomState(31)
    y = rng.randint(0, 2, n)
    jets['eventNumber'] = events
    jets['HadronConeExclTruthLabelID'] = 5 * y
    jets['pt_btagJes'] = 20000 + rng.uniform(0, 250000, n)
    jets['eta_btagJes'] = rng.uniform(-2.4, 2.4, n)
    for prefix, strength in [('GN2v01', 2.), ('DL1dv01', 1.5)]:
        p = 1 / (1 + np.exp(-(strength * (2 * y - 1) + rng.normal(size=n))))
        jets[prefix + '_pb'] = p
        jets[prefix + '_pc'] = (1 - p) * .2
        if prefix == 'GN2v01':
            jets[prefix + '_pu'] = (1 - p) * .7
            jets[prefix + '_ptau'] = (1 - p) * .1
        else:
            jets[prefix + '_pu'] = (1 - p) * .8
    tracks = np.zeros((n, 40), dtype=[(k, '?' if k == 'valid' else 'f4') for k in sorted(required['tracks'])])
    tracks['valid'][:, :4] = True
    for name in required['tracks'] - {'valid'}:
        tracks[name][:, :4] = rng.uniform(.01, 1, (n, 4))
    tracks['lifetimeSignedD0Significance'][:, :4] += y[:, None] * 2
    h5 = directory / 'software_fixture.h5'
    with h5py.File(h5, 'w') as handle:
        handle.create_dataset('jets', data=jets, chunks=(20,))
        handle.create_dataset('tracks', data=tracks, chunks=(20, 40))
        handle.create_dataset('eventwise', data=np.zeros(n))
        handle.create_dataset('truth_hadrons', data=np.zeros(n))
    encoder = PhysicsConstrainedGatedTrackGeometry(hidden_dimension=8, geometry_dimension=6, max_tracks=4, graph_neighbors=2,
        sequential_steps=1, max_epochs=2, patience=2, batch_size=20, random_state=3)
    model = DerivativeGatedNeuralPOR(n_neighbors=6, scale_neighbor=3, query_batch_size=16, denominator_floor=1e-12)
    model.selection_training_indices_ = np.arange(32)
    model.measurement_training_indices_ = np.arange(60)
    bundle = {'sampled_source_rows': source_rows, 'sampled_event_numbers': events, 'neural_por_model': model,
              'independent_test_indices': splits['independent_test']}
    _, _, roles, measurement = validate_splits(bundle, splits)
    adapter = IdentityFeatureMap(6)
    with H5StructuredTrackSource(h5, source_rows, max_tracks=4) as source:
        encoder.fit(source, splits['encoder_train'], y[splits['encoder_train']], splits['encoder_early_stop'], y[splits['encoder_early_stop']])
        e = encoder.encode_source(source, base, batch_size=20)['geometry']
        model.neural_geometry_ = NeuralGeometryMap(max_epochs=2, patience=2, architecture='shallow', batch_size=20).fit(e, y[base])
        metric = model.neural_geometry_.transform(e)
        graph_base = model._prepare_base(metric[:32], y[base[:32]])
        model.core_ = model._fit_operator(graph_base, .1, 3)
        model.candidate_bank_k_ = 3
        model.selected_orbital_indices_ = np.array([0, 1, 2])
        model.selected_k_ = 3
        model.required_orbital_bank_ = 3
        model.label_encoder_ = LabelEncoder().fit([0, 1])
        train_e = encoder.encode_source(source, measurement, batch_size=20)['geometry']
        model.A_train_ = model.transform(adapter.transform(train_e))
        model.y_train_encoded_ = y[measurement]
        model.measurement_ = StructuredOrbitalMeasurement(max_iter=40, n_restarts=1).fit(model.A_train_, y[measurement])
        test_e = encoder.encode_source(source, roles['independent_test'], batch_size=20)['geometry']
        p = model.predict_proba(adapter.transform(test_e))[:, 1]
    expected = run_paper.expected_frame(source_rows, events, roles['independent_test'], y[roles['independent_test']])
    expected.assign(self_configuring_por_probability_b=p).to_csv(model_dir / run_paper.DG_SCORES, index=False, compression='gzip')
    (model_dir / 'test_scoring_status.json').write_text(json.dumps({'scored': True}))
    integrity = {'tier': 'small', 'file_name': h5.name, 'size_bytes': h5.stat().st_size, 'adler32': file_adler32(h5), 'TEST_FIXTURE_ONLY': True}
    bundle.update(structured_track_encoder=encoder, input_feature_adapter=adapter,
        analysis_protocol='paper-ttbar', label_definition='cone', label_field='HadronConeExclTruthLabelID', dataset_tier='small',
        method_resolved_before_test=True, official_file_size=integrity['size_bytes'], official_file_adler32=integrity['adler32'],
        source_tree_sha256=source_identity(ROOT)['source_tree_sha256'], sample=n, max_tracks=4,
        selected_architecture={'name': 'SOFTWARE_FIXTURE_ONLY'})
    joblib.dump(bundle, model_dir / run_paper.DG_MODEL)
    pn_dir = directory / 'particlenet'
    train_particle_net(model_dir, h5, pn_dir, architecture='full', preset='quick', epochs=1, device='cpu', batch_size=32, scaler_sample=32, max_tracks=4)
    wp = roles['wp_validation']
    expected_wp = run_paper.expected_frame(source_rows, events, wp, y[wp])
    return {'directory': directory, 'model_dir': model_dir, 'h5': h5, 'pn_dir': pn_dir, 'bundle': bundle,
            'splits': splits, 'roles': roles, 'integrity': integrity, 'expected_test': expected, 'expected_wp': expected_wp}


def test_existing_particle_net_alignment_and_manifest(experiment):
    e = experiment
    manifest, aligned, info = run_paper.validate_particle(e['pn_dir'], e['expected_test'], e['expected_wp'], e['roles'], e['bundle'])
    assert manifest['architecture'] == 'full'
    assert len(aligned) == len(e['expected_test'])
    assert info['source_split_verification'] == 'exact source split arrays verified'
    assert not info['legacy_track_limit_assumed_20']


def test_actual_h5_fixture_is_rejected_by_production_integrity(experiment):
    with pytest.raises(RuntimeError, match='size mismatch'):
        verify_official_file(experiment['h5'], tier='small')


def test_wrong_prediction_rows_or_labels_rejected(experiment):
    expected = experiment['expected_test']
    frame = expected.assign(p=np.linspace(.1, .9, len(expected)))
    frame.loc[0, 'y_true_light0_b1'] = 1 - frame.loc[0, 'y_true_light0_b1']
    with pytest.raises(ValueError, match='match'):
        run_paper.align_predictions(frame, expected, 'p')
    frame = expected.assign(p=.5)
    frame.loc[1, 'source_row'] = frame.loc[0, 'source_row']
    with pytest.raises(ValueError, match='unique'):
        run_paper.align_predictions(frame, expected, 'p')


def test_full_paper_pipeline_and_cache(experiment, monkeypatch):
    e = experiment
    import por_hep.atlas_jetset as atlas
    monkeypatch.setattr(atlas, 'verify_official_file', lambda *args, **kwargs: e['integrity'])
    output = e['directory'] / 'RESULTS_SOFTWARE_FIXTURE_NOT_ATLAS'
    before = sha256(e['model_dir'] / run_paper.DG_MODEL)
    original_source = source_identity(ROOT)
    argv = ['--model-dir', str(e['model_dir']), '--h5-file', str(e['h5']), '--particle-net-dir', str(e['pn_dir']),
            '--output', str(output), '--bootstrap', '20', '--read-batch-size', '20']
    assert run_paper.main(argv) == 0
    assert before == sha256(e['model_dir'] / run_paper.DG_MODEL)
    assert source_identity(ROOT) == original_source
    table = pd.read_csv(output / 'paper/TABLE_1_COMPARISON.csv')
    assert list(table.method) == ['DG-NPOR', 'ParticleNet (JetSet)', 'GN2v01', 'DL1dv01']
    assert np.isfinite(table.auc).all()
    metadata = json.loads((output / 'COMPARISON_METADATA.json').read_text())
    assert metadata['K_DG'] == 3 and not metadata['future_work_implemented']
    assert metadata['dg_score_receipt']['source'].startswith('original stored')
    assert (output / 'PAPER_RESULTS.zip').is_file()
    assert (output / 'OVERLEAF_RESULTS.zip').is_file()
    assert (output / 'figures/FIGURE_1_REJECTION.pdf').is_file()
    assert (output / 'paper/RESULTS_PREVIEW.pdf').is_file()
    # Re-exporting the same frozen run must not train or reopen track tensors.
    def forbidden_read(*args, **kwargs):
        raise AssertionError('Cached report re-opened tracks.')
    monkeypatch.setattr(H5StructuredTrackSource, 'read', forbidden_read)
    assert run_paper.main(argv) == 0
    assert before == sha256(e['model_dir'] / run_paper.DG_MODEL)
    if os.environ.get('DG_PAPER_QA_LOCATION'):
        Path(os.environ['DG_PAPER_QA_LOCATION']).write_text(str(output))


def test_missing_completed_baseline_fails_instead_of_publishing_three_methods(experiment, monkeypatch, tmp_path):
    e = experiment
    monkeypatch.setattr(run_paper, 'particle_candidates', lambda *a: [])
    args = run_paper.parse_args([])
    with pytest.raises(FileNotFoundError, match='ParticleNet'):
        run_paper.get_particle(args, {}, tmp_path, e['model_dir'], e['h5'], e['expected_test'], e['expected_wp'], e['roles'], e['bundle'])


def test_censored_rejection_table_has_lower_bound():
    from paper_report import rejection_latex
    text = rejection_latex({'R70': np.inf, 'R70_lower68': 100., 'R70_upper68': np.inf}, 'R70')
    assert '\\infty' in text and '100.0' in text


def test_changed_context_does_not_reuse_old_results(experiment, monkeypatch, tmp_path):
    e = experiment
    import por_hep.atlas_jetset as atlas
    monkeypatch.setattr(atlas, 'verify_official_file', lambda *a, **kw: e['integrity'])
    (tmp_path / 'RUN_CONTEXT.json').write_text(json.dumps({'dg_model_sha256': 'different'}))
    with pytest.raises(RuntimeError, match='different frozen experiment'):
        run_paper.main(['--model-dir', str(e['model_dir']), '--h5-file', str(e['h5']), '--output', str(tmp_path), '--bootstrap', '20'])
