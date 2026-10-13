# Concept-generation variance: BLOCKED

## Missing requirement

DEEPSEEK_API_KEY is not available in the process environment or repository .env.

No LLM was substituted and no replicate was fabricated.

## Exact resume command

cd /workspace/Label-free-CBM-Audio && test -n "$DEEPSEEK_API_KEY" && /venv/main/bin/python scripts/concepts/generation_variance.py --repo-root . --generate

The command reads DEEPSEEK_API_KEY from the process environment only; the value must not be printed or committed.

## Expected output locations

- data/concept_sets/esc50/revision_variance/broad/replicate_01/
- data/concept_sets/esc50/revision_variance/broad/replicate_02/
- data/concept_sets/esc50/revision_variance/contrastive/replicate_01/
- data/concept_sets/esc50/revision_variance/contrastive/replicate_02/
- results/revision/t8_checks/generation_variance/comparison_summary.json
- results/revision/t8_checks/generation_variance/comparison.md
