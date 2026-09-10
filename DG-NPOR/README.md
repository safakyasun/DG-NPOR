# DG-NPOR

Predictive orbital representations for **b-jet versus light-jet classification** using ATLAS JetSet simulation data.

This repository contains the DG-NPOR source code, a physics-constrained learned geometry, derivative-gated orbital selection, a positive semidefinite (PSD) measurement readout, and scripts for evaluating a saved model against a local ParticleNet adaptation and the GN2v01/DL1dv01 reference scores supplied with JetSet.

The selected orbital count is data dependent. The code does not impose three orbitals on every dataset or training run.

## Source layout

| Path | Contents |
| --- | --- |
| `engines/dg_npor/src/por_core/` | Graph, information potential, operator, orbitals, dimension selection and measurement readout |
| `engines/dg_npor/src/por_hep/` | JetSet loading, structured tracks, learned geometry and evaluation utilities |
| `engines/dg_npor/run_self_configuring_nn.py` | DG-NPOR development/training entry point |
| `engines/dg_npor/scripts/` | Orbital and workflow figures |
| `engines/paper/src/jetset_litcomp/` | ParticleNet adaptation, training and metrics |
| `run_paper.py` | Saved-model evaluation entry point |
| `configs/paths.json` | Legacy input-location candidates; explicit command-line paths take priority |
| `verification/` | Software checks and original preparation-stage validation record |

Keep this directory structure intact: the entry points resolve the bundled source directories automatically. The frozen-model evaluator also checks the DG-NPOR source identity against the saved experiment.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

ParticleNet training additionally requires PyTorch:

```bash
python -m pip install -r requirements_training.txt
```

The requirements specify compatible version ranges, not a fully pinned reproduction environment. Compatibility with a particular saved model must be checked against its original environment.

## Data and saved artifacts

The evaluation uses the ATLAS JetSet `mc-flavtag-ttbar-small.h5` file. Dataset download details are in `engines/dg_npor/download_atlas_jetset.py` and `engines/dg_npor/src/por_hep/atlas_jetset.py`.

The repository contains source code; the dataset, trained weights, prediction tables and final experimental outputs are not bundled.

To evaluate an existing DG-NPOR experiment, retain at least these files together in its model directory:

- `trained_self_configuring_dg_npor_v9.joblib`
- `frozen_split_indices.npz`

To reuse a compatible completed ParticleNet experiment, provide its output directory containing:

- `particle_net_manifest.json`
- `particle_net_wp_validation_predictions.csv.gz`
- `particle_net_locked_test_predictions.csv.gz`

The evaluator checks model/source identity, dataset identity, split roles and prediction alignment. It rejects incompatible artifacts.

## Evaluate saved models

Run from the repository root, replacing the example absolute paths with the actual locations:

```bash
python run_paper.py \
  --model-dir /absolute/path/to/dg_model_directory \
  --h5-file /absolute/path/to/mc-flavtag-ttbar-small.h5 \
  --particle-net-dir /absolute/path/to/completed_particlenet_directory \
  --output outputs/paper500k
```

This command retains the saved DG-NPOR model and its selected orbitals. It does not retrain DG-NPOR. To allow training the local ParticleNet baseline when no compatible completed run is available, add `--train-particle-net-if-missing`. The `run_paper.sh` wrapper enables that option automatically.

Use `python run_paper.py --help` for all options.

## Train a new DG-NPOR experiment

Inspect the development entry point before choosing the sample size and preset:

```bash
python engines/dg_npor/run_self_configuring_nn.py --help
```

For example, a new quick development run can use:

```bash
python engines/dg_npor/run_self_configuring_nn.py \
  --preset quick \
  --h5-file /absolute/path/to/mc-flavtag-ttbar-small.h5 \
  --output-dir outputs/dg_npor_quick
```

This launches training. A quick run is a separate experiment and is not a reproduction of the saved 500,000-jet experiment. Exact reproduction requires the original configuration, split artifacts, model and software environment.

## Evaluation notes

- All comparison predictions are aligned to the same evaluation jets and labels.
- GN2v01 and DL1dv01 are pretrained references; this is not a matched training-budget comparison.
- ParticleNet is the bundled JetSet adaptation, not a claim of exact reproduction of the published implementation.
- Working-point rejection uses the signal-score quantiles in the evaluation sample.
- The included validation records concern software fixtures, not ATLAS performance measurements.

## Software verification

The existing integration checks use small synthetic HDF5 fixtures. To run them:

```bash
python -m pip install -r requirements_training.txt pytest
python -m pytest verification/test_paper.py -q
```

The full fixture integration requires PyTorch. These checks do not validate the physics performance of a trained model.
