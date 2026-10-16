#!/usr/bin/env bash
set -euo pipefail

repo_root="/workspace/Label-free-CBM-Audio"
run_dir="${repo_root}/results/revision/esc50_compressed_gpu1"
fold2_pid="${1:?fold-2 runner PID is required}"
fold2_complete="${run_dir}/fold2/fold_complete.json"

while kill -0 "${fold2_pid}" 2>/dev/null; do
  if [[ -f "${fold2_complete}" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -f "${fold2_complete}" ]]; then
  echo "fold 2 did not produce a completion marker; refusing to start later folds" >&2
  exit 1
fi

for fold in 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 \
  HF_HOME=/workspace/.hf_home_task4 \
  TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /venv/main/bin/python -m scripts.evaluation.run_esc50_compressed \
    --repo-root "${repo_root}" \
    --run-id esc50_compressed_gpu1 \
    --fold "${fold}" \
    --device cuda \
    --num-workers 1 \
    --ast-batch-size 8 \
    --feature-batch-size 4 \
    >"${run_dir}/fold${fold}.launch.log" 2>&1
done
