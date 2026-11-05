#!/bin/bash
# Historical: the driver of the first MS MARCO validation, kept to show how those records were made. Current experiments use scripts/steps.sh.
# Phase 0 close-out validation on MS MARCO passage (dev queries).
#
# Tests the shipped default engine where Phase 0 has no evidence yet: real short web queries, untitled passages
# (lead proxies: the case where Stage 2 can hurt with no label-free signal noticing), and a 1M-passage index.
# Long (several hours, dominated by encoding 1M passages with five models); run it alone on the GPU:
#
#     nohup scripts/closeout_experiments.sh > outputs/closeout/driver.log 2>&1 &
#
# Steps run one at a time (rerank jobs must not overlap: the rerank score cache is one file per reranker).
# A finished step leaves outputs/closeout/<step>.done and is skipped on a rerun, so the script can be restarted.
# Embeddings, anchors and reranker scores are cached under data/cache/, so reruns of a step are fast.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/closeout
mkdir -p "$OUT"

step() {  # name, then spacebridge-bench arguments
  local name=$1
  shift
  if [ -f "$OUT/$name.done" ]; then
    echo "skip $name (done)"
    return
  fi
  echo "start $name $(date '+%F %T')"
  rm -f "$OUT/$name.jsonl"  # a step that did not finish is rerun from scratch
  uv run python -u -m spacebridge_bench.run "$@" --out "$OUT/$name.jsonl" > "$OUT/$name.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch "$OUT/$name.done"
  echo "end $name exit $rc $(date '+%F %T')"
  [ $rc -eq 0 ] || failed="$failed $name"
}
failed=""

SEEDS="0 1 2"

# 1. 50K-passage subsample (comparable to the other Phase 0 sets): first stage, ablation rows, 3 seeds.
#    --stage2-curves scores every Stage-2 checkpoint on the test queries, so scripts/stage2_rules.py can check
#    whether Stage 2 helps or hurts on untitled short queries (Phase 0 close-out risk 1).
step ms50k_main --dataset msmarco --tier 1 3 --rows 0a 0b 1 6 8 8b --K 2048 4096 --seed $SEEDS --stage2-curves

# 2. 50K: end to end (bridged top-30 / 60 / 100 + bge-reranker-v2-m3), default engine, K = 2,048.
step ms50k_rerank --dataset msmarco --tier 1 3 --rows 0a 0b 8b 9n 9 9w --K 2048 --seed $SEEDS --rerankers bge-reranker-v2-m3

# 3. 1M-passage index (scale test): first stage, 3 seeds.
step ms1m_main --dataset msmarco-1m --tier 1 3 --rows 0a 0b 1 6 8 8b --K 2048 4096 --seed $SEEDS

# 4. 1M: model-upgrade scenario. An index built with an older, smaller model, queried with current ones.
step ms1m_upgrade --dataset msmarco-1m --pair minilm:bge-large-en minilm:qwen3-0.6b bge-base-en:qwen3-0.6b \
  --rows 0a 0b 1 6 8 8b --K 2048 4096 --seed $SEEDS

# 5. 1M: end to end, default engine, K = 2,048, one seed (rerank cost grows with queries, not corpus size).
step ms1m_rerank --dataset msmarco-1m --tier 1 3 --rows 0a 0b 8b 9n 9 9w --K 2048 --seed 0 --rerankers bge-reranker-v2-m3

# 6. 1M: mount and solver time on the default protocol. Must run alone on the GPU (timings under contention
#    are meaningless); it is last so nothing else from this script overlaps it.
step ms1m_mount --dataset msmarco-1m --tier 1 3 --pair minilm:bge-large-en minilm:qwen3-0.6b --rows 8 8b --K 2048 \
  --seed 0 --time-mount

if [ -n "$failed" ]; then
  echo "FAILED:$failed $(date '+%F %T')"
  exit 1
fi
echo "ALL DONE $(date '+%F %T')"
