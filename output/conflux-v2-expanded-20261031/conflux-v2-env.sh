export STUDY_ROOT=/mnt/data/mdevos/conflux-v2-20261031
export TMPDIR=$STUDY_ROOT/tmp
export XDG_CACHE_HOME=$STUDY_ROOT/cache
export HF_HOME=$STUDY_ROOT/cache/huggingface
export MPLCONFIGDIR=$STUDY_ROOT/cache/matplotlib
export TORCH_HOME=$STUDY_ROOT/cache/torch
export TORCHINDUCTOR_CACHE_DIR=$STUDY_ROOT/cache/inductor
export PIP_CACHE_DIR=/mnt/data/mdevos/cache/pip
export CONDA_PKGS_DIRS=/mnt/data/mdevos/conda/pkgs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH=/mnt/data/mdevos/conda/envs/conflux-gpu/bin:$PATH
export PYTHONUNBUFFERED=1
cd "$STUDY_ROOT/code-study"
