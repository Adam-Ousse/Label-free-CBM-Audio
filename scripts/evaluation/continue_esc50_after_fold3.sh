#!/usr/bin/env bash
set -euo pipefail

repo_root="/workspace/Label-free-CBM-Audio"
run_dir="${repo_root}/results/revision/esc50_compressed_gpu1"
runner_pid="${1:?current ESC runner PID is required}"

for fold in 3 4 5; do
  marker="${run_dir}/fold${fold}/fold_complete.json"
  if [[ "${fold}" == "3" ]]; then
    while kill -0 "${runner_pid}" 2>/dev/null; do
      sleep 30
    done
    if [[ ! -f "${marker}" ]]; then
      echo "fold ${fold} runner exited without a completion marker" >&2
      exit 1
    fi
  elif [[ ! -f "${marker}" ]]; then
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
  fi
done
