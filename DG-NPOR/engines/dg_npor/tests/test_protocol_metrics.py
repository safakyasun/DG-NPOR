import numpy as np

from por_hep.metrics import stratified_bootstrap_intervals
from por_hep.protocol import (
    fixed_size_role_indices,
    locked_split_indices,
    stratified_refit_indices,
)


def test_official_tail_and_fixed_roles_are_auditable_and_disjoint():
    y = np.tile([0, 1], 600)
    locked = locked_split_indices(
        y, protocol="uci-official", official_test_rows=200
    )
    np.testing.assert_array_equal(locked.test, np.arange(1000, 1200))
    roles = fixed_size_role_indices(
        locked.development, y, 300, 200, 250, random_state=10
    )
    assert len(roles.base) == 300
    assert len(roles.operator_validation) == 200
    assert len(roles.dimension_audit) == 250
    assert not np.intersect1d(roles.base, roles.operator_validation).size
    assert not np.intersect1d(roles.base, roles.dimension_audit).size
    assert not np.intersect1d(
        roles.operator_validation, roles.dimension_audit
    ).size
    refit = stratified_refit_indices(
        locked.development, y, size=400, random_state=11
    )
    assert len(refit) == 400


def test_bootstrap_intervals_report_cap_and_contain_finite_values():
    rng = np.random.RandomState(13)
    y = np.tile([0, 1], 500)
    probability = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.12, len(y)), 0.001, 0.999)
    report = stratified_bootstrap_intervals(
        y,
        probability,
        repetitions=30,
        max_samples=400,
        random_state=14,
    )
    assert report["locked_test_rows"] == 1000
    assert report["rows_per_replicate"] == 400
    for interval in report["intervals"].values():
        assert np.isfinite(interval["estimate"])
        assert interval["lower"] <= interval["upper"]
