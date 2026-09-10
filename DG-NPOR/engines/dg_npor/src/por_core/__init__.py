"""Derivative-gated Neural Predictive Orbital Representation."""

from .estimator_derivative_gated import DerivativeGatedNeuralPOR
from .neural_geometry import NeuralGeometryMap
from .derivative_gated_dimension import (
    DerivativeGatedDimensionResult,
    select_derivative_gated_dimension,
)
from .predictive_field_dimension import (
    PredictiveFieldDimensionResult,
    select_predictive_field_dimension,
)
from .working_point_scan import (
    degeneracy_safe_ranked_subset,
    eventwise_development_split,
    fit_structured_readout,
)

__all__ = [
    "DerivativeGatedNeuralPOR",
    "NeuralGeometryMap",
    "DerivativeGatedDimensionResult",
    "select_derivative_gated_dimension",
    "PredictiveFieldDimensionResult",
    "select_predictive_field_dimension",
    "degeneracy_safe_ranked_subset",
    "eventwise_development_split",
    "fit_structured_readout",
]

__version__ = "9.2.0-wp-scan"
