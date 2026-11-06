#!/bin/bash
# The review's two follow-ups: an interval for the fresh HNSW check, and whether CPU reranking agrees with the GPU's. Restartable.
OUT=outputs/review/followup; . scripts/steps.sh
export PYTHONPATH=$PWD/src
PY=.venv/bin/python
failed=""
FULL_INDEX=$(ls data/cache/hnsw/msmarco-full_*.faiss | head -1)
step hnsw_per_query $PY -u scripts/hnsw_study.py msmarco-full-fresh --index-model bge-large-en --query-models bge-base-en qwen3-0.6b --index-path "$FULL_INDEX" --pools 60 --efs 500 --lexical --per-query "$OUT/hnsw_per_query.npz" --out "$OUT/hnsw.jsonl" || failed="$failed hnsw_per_query"
# The candidates of the earlier precision check (the index model's own top 60), plus long documents. GPU fp32 and bf16 come from cache.
for spec in "msmarco-1m 100" "nq-1m 100" "fiqa 50" "trec-covid 50"; do
  set -- $spec
  step "cpu_$1" $PY -u scripts/check_reranker_precision.py $1 --queries $2 --pools 60 --variants fp32 bf16 cpu --out "$OUT/precision_cpu.jsonl" || failed="$failed cpu_$1"
done
if [ -n "$failed" ]; then echo "FOLLOWUP FAILED:$failed"; exit 1; fi
echo "FOLLOWUP DONE $(date '+%F %T')"
