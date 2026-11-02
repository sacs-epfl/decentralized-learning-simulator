export STUDY_ROOT=/mnt/data/mdevos/conflux-fedopt-20261102
export TMPDIR=$STUDY_ROOT/tmp
export XDG_CACHE_HOME=$STUDY_ROOT/cache
export HF_HOME=/mnt/data/mdevos/conflux-v2-20261031/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR=$STUDY_ROOT/cache/matplotlib
export TORCH_HOME=$STUDY_ROOT/cache/torch
export TORCHINDUCTOR_CACHE_DIR=$STUDY_ROOT/cache/inductor
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH=/mnt/data/mdevos/conda/envs/conflux-gpu/bin:$PATH
export PYTHONUNBUFFERED=1
cd "$STUDY_ROOT/code"
