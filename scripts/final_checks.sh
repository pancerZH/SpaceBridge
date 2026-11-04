#!/bin/bash
# Checks for the decisions accepted on 2026-11-04. Restartable: rerun the same command and it continues from what is on disk.
#   Real mounts at K = 8,192's line, 160,001 MS MARCO documents.
#   Both sides of that line with BM25: K = 2,048 at 160,000 and K = 8,192 at 160,001, on three corpora.
#   The real Searcher on HotpotQA's 50K set, below BM25's old line, against row 9h.
# Results go to outputs/phase1/final/. Run from the checkout whose src is the code under test.
OUT=outputs/phase1/final; . scripts/steps.sh
export PYTHONPATH=$PWD/src
PY=.venv/bin/python
RR="--rerankers bge-reranker-v2-m3"
MS="--pair bge-large-en:bge-base-en bge-large-en:qwen3-0.6b harrier-0.6b:bge-large-en"
NQ="--pair bge-large-en:bge-base-en bge-large-en:qwen3-0.6b qwen3-0.6b:bge-large-en"
R="--rows 0a 0b 8b 9 9h"
run() { local name=$1; shift; step "$name" $PY -u -m spacebridge_bench.run "$@" --out "$OUT/$name.jsonl"; }
set -e
run budget160k --dataset msmarco-1m-nested --max-docs 160001 --max-queries 20 --pair bge-large-en:qwen3-0.6b bge-large-en:bge-base-en bge-base-en:bge-large-en harrier-0.6b:bge-large-en --rows 0a 0b 8b --K 0 --seed 0 --time-mount
for n in 160000 160001; do
  run ms$n --dataset msmarco-1m-nested --max-docs $n $MS $R --max-queries 500 --K 0 --seed 0 1 2 $RR
  run nq$n --dataset nq-1m-nested --max-docs $n $NQ $R --max-queries 500 --K 0 --seed 0 1 2 $RR
  run hp$n --dataset hotpotqa-1m-nested --max-docs $n $NQ $R --max-queries 500 --K 0 --seed 0 1 2 $RR
done
run searcher_row --dataset hotpotqa --pair bge-large-en:bge-base-en --rows 0a 0b 8b 9 9h --K 0 --seed 0 --max-queries 200 $RR
step searcher_check $PY -u scripts/searcher_check.py hotpotqa --index-model bge-large-en --query-model bge-base-en --n 200 --budgets 60 --bench-fit --same-anchors --out "$OUT/searcher_check.jsonl"
echo "FINAL CHECKS DONE $(date '+%F %T')"
