import numpy as np

from compare_locked_test_baselines import (
    PAPER_WORKING_POINTS,
    _paper_working_point_rows,
)


def test_paper_working_points_exclude_auc_and_average_precision():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95])
    rows = _paper_working_point_rows("paper-test", y, score, "unit-test")

    assert len(rows) == len(PAPER_WORKING_POINTS)
    assert [row["target_b_efficiency"] for row in rows] == list(
        PAPER_WORKING_POINTS
    )
    for row in rows:
        assert "auc" not in row
        assert "average_precision" not in row
        assert "light_rejection" in row
        assert row["confidence_interval"] == "two-sided Clopper-Pearson 68%"
