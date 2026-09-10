import numpy as np

from por_hep import CompactATLASPhysicsFeatureMap


HIGGS_FEATURE_NAMES = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
    "PRI_tau_pt",
    "PRI_tau_eta",
    "PRI_tau_phi",
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_met",
    "PRI_met_phi",
    "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]


def _sample(seed=17, rows=80):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(rows, len(HIGGS_FEATURE_NAMES)))
    for name in CompactATLASPhysicsFeatureMap.OBJECT_PHI + (
        CompactATLASPhysicsFeatureMap.REFERENCE_PHI,
    ):
        X[:, HIGGS_FEATURE_NAMES.index(name)] = rng.uniform(-np.pi, np.pi, rows)
    return X


def test_clean_higgs_map_is_30_to_33_and_global_azimuth_invariant():
    X = _sample()
    feature_map = CompactATLASPhysicsFeatureMap(HIGGS_FEATURE_NAMES).fit(X)
    baseline = feature_map.transform(X)
    assert baseline.shape == (len(X), 33)

    rotated = X.copy()
    event_rotation = np.linspace(-2.3, 3.1, len(X))
    for name in CompactATLASPhysicsFeatureMap.OBJECT_PHI + (
        CompactATLASPhysicsFeatureMap.REFERENCE_PHI,
    ):
        rotated[:, HIGGS_FEATURE_NAMES.index(name)] += event_rotation
    np.testing.assert_allclose(
        baseline, feature_map.transform(rotated), rtol=0.0, atol=2e-14
    )


def test_missingness_is_one_flag_per_original_training_column():
    X = _sample(seed=19)
    X[0:5, HIGGS_FEATURE_NAMES.index("DER_mass_MMC")] = -999.0
    X[5:10, HIGGS_FEATURE_NAMES.index("PRI_jet_subleading_phi")] = -999.0
    feature_map = CompactATLASPhysicsFeatureMap(HIGGS_FEATURE_NAMES).fit(X)
    transformed = feature_map.transform(X)

    assert transformed.shape[1] == 35
    assert feature_map.missing_feature_names_ == [
        "missing__DER_mass_MMC",
        "missing__PRI_jet_subleading_phi",
    ]
    assert set(np.unique(transformed[:, -2:])).issubset({0.0, 1.0})
    assert np.isfinite(transformed).all()
