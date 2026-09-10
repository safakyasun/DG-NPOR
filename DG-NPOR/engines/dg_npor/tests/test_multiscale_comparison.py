import numpy as np
import pandas as pd

from compare_multiscale_to_graph_por import main


def test_paired_multiscale_comparison_uses_identical_source_rows(tmp_path):
    new_output = tmp_path / "new"
    old_output = tmp_path / "old"
    new_output.mkdir()
    old_output.mkdir()
    rows = np.arange(40)
    labels = np.repeat([0, 1], 20)
    old_score = np.concatenate(
        [np.linspace(0.05, 0.80, 20), np.linspace(0.30, 0.98, 20)]
    )
    new_score = old_score.copy()
    new_score[:20] -= np.linspace(0.0, 0.20, 20)
    for output, score in ((new_output, new_score), (old_output, old_score)):
        pd.DataFrame(
            {
                "source_row": rows,
                "y_true_light0_b1": labels,
                "structured_por_probability_b": score,
            }
        ).to_csv(
            output / "structured_locked_test_predictions.csv.gz", index=False
        )
    assert main([str(new_output), str(old_output)]) == 0
    comparison = pd.read_csv(new_output / "paired_multiscale_vs_graph_70op.csv")
    paired = pd.read_csv(
        new_output / "paired_multiscale_vs_graph_mistag_test.csv"
    )
    assert list(comparison["method"]) == [
        "Selected multiscale POR",
        "Previous one-step Graph-POR",
    ]
    assert paired.loc[0, "discordant_light_jets"] >= 0
    assert 0.0 <= paired.loc[0, "exact_paired_binomial_p_value"] <= 1.0

