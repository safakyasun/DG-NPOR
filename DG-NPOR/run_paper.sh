#!/usr/bin/env bash
set -euo pipefail
DG_PAPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${DG_PAPER_PYTHON:-}" ]]; then
    DG_PAPER_INTERPRETER="$DG_PAPER_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    DG_PAPER_INTERPRETER="$CONDA_PREFIX/bin/python"
else
    DG_PAPER_INTERPRETER="$(command -v python3 || true)"
fi
if [[ -z "$DG_PAPER_INTERPRETER" ]]; then
    echo 'Önce conda activate z-mumu-por çalıştır.' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTORCH_ENABLE_MPS_FALLBACK=1
if ! "$DG_PAPER_INTERPRETER" - <<'PY'
import importlib
import sys
if sys.version_info < (3, 10):
    sys.exit('Python >=3.10 gerekli. Aktif yorumlayıcı: ' + sys.executable)
missing = []
for name in ('numpy', 'scipy', 'sklearn', 'pandas', 'h5py', 'joblib', 'matplotlib'):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    sys.exit('Eksik paketler: ' + ', '.join(missing))
print('Kullanılan Python:', sys.executable, flush=True)
PY
then
    printf 'Kurulum: "%s" -m pip install -r "%s/requirements.txt"\n' "$DG_PAPER_INTERPRETER" "$DG_PAPER_DIR" >&2
    exit 1
fi
exec "$DG_PAPER_INTERPRETER" "$DG_PAPER_DIR/run_paper.py" --train-particle-net-if-missing "$@"
