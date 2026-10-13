#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

exec /venv/main/bin/python -m scripts.evaluation.task4_cv \
  --repo-root "${repo_root}" \
  "$@"
