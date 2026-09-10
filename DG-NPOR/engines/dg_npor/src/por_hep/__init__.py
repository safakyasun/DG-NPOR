"""Dataset adapters and locked evaluation protocols used by POR V9."""

from .data import ParticleDataset, load_particle_dataset
from .densenet_features import DenseNet201DirectFeatureMap
from .higgs_compact_features import CompactATLASPhysicsFeatureMap
from .physics_features import PhysicsInvariantFeatureMap
from .susy_features import SUSYPhysicsFeatureMap, SUSY_FEATURE_NAMES
from .atlas_jetset import (
    AtlasJetSetFeatureMap,
    JETSET_FILES,
    JETSET_PROTOCOLS,
    dl1d_paper_b_discriminant,
    download_official_file,
    gn2_paper_b_discriminant,
    load_atlas_jetset_b_vs_light,
)
from .structured_tracks import (
    AtlasJetSetIndex,
    H5StructuredTrackSource,
    IdentityFeatureMap,
    StructuredTrackGeometry,
    load_atlas_jetset_index,
)
from .self_configuring_geometry import (
    PhysicsConstrainedGatedTrackGeometry,
    dropout_stability_audit,
)
from .protocol import (
    LockedSplitIndices,
    RoleIndices,
    fixed_size_role_indices,
    grouped_locked_split_indices,
    locked_split_indices,
    stratified_group_holdout_indices,
    stratified_refit_indices,
)

__all__ = [
    "ParticleDataset",
    "load_particle_dataset",
    "DenseNet201DirectFeatureMap",
    "CompactATLASPhysicsFeatureMap",
    "PhysicsInvariantFeatureMap",
    "SUSYPhysicsFeatureMap",
    "SUSY_FEATURE_NAMES",
    "AtlasJetSetFeatureMap",
    "JETSET_FILES",
    "JETSET_PROTOCOLS",
    "gn2_paper_b_discriminant",
    "dl1d_paper_b_discriminant",
    "download_official_file",
    "load_atlas_jetset_b_vs_light",
    "AtlasJetSetIndex",
    "H5StructuredTrackSource",
    "IdentityFeatureMap",
    "StructuredTrackGeometry",
    "load_atlas_jetset_index",
    "PhysicsConstrainedGatedTrackGeometry",
    "dropout_stability_audit",
    "LockedSplitIndices",
    "RoleIndices",
    "fixed_size_role_indices",
    "grouped_locked_split_indices",
    "locked_split_indices",
    "stratified_group_holdout_indices",
    "stratified_refit_indices",
]
