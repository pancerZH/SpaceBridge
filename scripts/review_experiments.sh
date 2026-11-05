#!/bin/bash
# The experiments of the review round (docs/fresh_confirmation_plan.md). Restartable: rerun it and finished steps are skipped. Stops starting steps,
# and cuts off any running one, 12 hours after its first start. Results go to outputs/review/.
OUT=outputs/review; . scripts/steps.sh
export PYTHONPATH=$PWD/src
PY=.venv/bin/python
LIMIT_S=$((12 * 3600 - 600))
[ -f "$OUT/started_at" ] || date +%s > "$OUT/started_at"
START=$(cat "$OUT/started_at")
failed=""; skipped=""
guard() {  # name, then the command: skipped when no time is left, cut off when it runs out
  local name=$1; shift
  [ -f "$OUT/$name.done" ] && return 0
  local left=$((START + LIMIT_S - $(date +%s)))
  if [ "$left" -le 60 ]; then skipped="$skipped $name"; echo "$(date '+%F %T') step $name: not started, out of time" | tee -a "$SPACEBRIDGE_PROGRESS"; return 0; fi
  step "$name" timeout --signal=INT "$left" "$@" || failed="$failed $name"
}
run() { local name=$1; shift; guard "$name" $PY -u -m spacebridge_bench.run "$@" --rerankers bge-reranker-v2-m3 --no-bridge --per-query-dir "$OUT/per_query/$name" --out "$OUT/$name.jsonl"; }
T1="bge-large-en:bge-base-en bge-base-en:bge-large-en"
T3="bge-large-en:qwen3-0.6b qwen3-0.6b:bge-large-en"
PAIRS="--pair $T1 $T3"
FRESH="--rows 0a 0b 8b 9 9h --K 0 --seed 0 1 2"

# 1. The published cells again, unchanged in configuration, keeping per-query scores. Everything is cached, so this is a regression check too.
$PY scripts/review_recompute.py --print-commands > "$OUT/recompute_commands.txt"
while read -r name args; do
  [ -n "$name" ] && run "recompute_$name" $args
done < "$OUT/recompute_commands.txt"

# 2. A selector study the review asked to repeat under the current defaults: anchor docs at 128 tokens.
for d in fiqa scidocs nq; do
  guard "stage1_$d" $PY -u scripts/stage1_study.py --dataset $d --out "$OUT/stage1_$d.jsonl"
done

# 3. Fresh confirmation, in the order the plan fixes.
run fresh_msmarco --dataset msmarco-full-fresh $PAIRS $FRESH
FULL_INDEX=$(ls data/cache/hnsw/msmarco-full_*.faiss | head -1)
guard fresh_msmarco_hnsw $PY -u scripts/hnsw_study.py msmarco-full-fresh --index-model bge-large-en --query-models bge-base-en qwen3-0.6b --index-path "$FULL_INDEX" --pools 60 --efs 500 --lexical --out "$OUT/fresh_msmarco_hnsw.jsonl"
run fresh_hotpotqa --dataset hotpotqa-full-fresh $PAIRS $FRESH

# 4. Latency, measured at the sizes the tables used to model. Serial, so the machine is idle apart from this.
QT="$PY -u scripts/query_timing.py"
CPU="env CUDA_VISIBLE_DEVICES="
C="--index-model bge-large-en --query-model qwen3-0.6b"
T="--out $OUT/timing.jsonl"
lex() { ls -d data/cache/lexical/$1_* | head -1; }
HNSW_1M=$(ls data/cache/hnsw/msmarco-1m_*_1024d_M32_efc200.faiss | grep de4f162f | head -1)
guard fit_100k $QT fit msmarco-1m-nested $C --max-docs 100000 --device cuda --bridge "$OUT/bridge_100k"
guard fit_1m $QT fit msmarco-1m $C --device cuda --bridge "$OUT/bridge_1m"
guard fit_2m $QT fit msmarco-full $C --max-docs 2000000 --device cuda --bridge "$OUT/bridge_2m"
guard fit_4m $QT fit msmarco-full $C --max-docs 4000000 --device cuda --bridge "$OUT/bridge_4m"
guard fit_full $QT fit msmarco-full $C --device cuda --bridge "$OUT/bridge_full"
guard fit_hp $QT fit hotpotqa-full $C --device cuda --bridge "$OUT/bridge_hp"
guard fit_tc $QT fit trec-covid $C --device cuda --bridge "$OUT/bridge_tc"
timing() {  # name, dataset, max-docs or -, bridge, backend, lexical, GPU search device or -, GPU queries, CPU queries
  local name=$1 ds=$2 md=$3 br=$4 be=$5 lx=$6 sd=$7 ng=$8 nc=$9
  local mdarg=""; [ "$md" != "-" ] && mdarg="--max-docs $md"
  local sdarg=""; [ "$sd" != "-" ] && sdarg="--search-device $sd"
  guard "t_${name}_gpu" $QT run $ds $C $mdarg --bridge "$OUT/$br" --pool 60 --backend "$be" --device cuda $sdarg --threads 10 --n $ng --lexical "$lx" --out "$OUT/timing_${name}_gpu.jsonl"
  guard "t_${name}_cpu" $CPU $QT run $ds $C $mdarg --bridge "$OUT/$br" --pool 60 --backend "$be" --device cpu --threads 10 --n $nc --lexical "$lx" --out "$OUT/timing_${name}_cpu.jsonl"
}
timing 100k msmarco-1m-nested 100000 bridge_100k exact "$(lex 'msmarco-1m-nested@100000')" - 200 30
timing 1m_hnsw msmarco-1m - bridge_1m "$HNSW_1M" "$(lex msmarco-1m)" - 200 30
timing 2m msmarco-full 2000000 bridge_2m exact "$(lex 'msmarco-full@2000000')" cpu 100 30
timing 4m msmarco-full 4000000 bridge_4m exact "$(lex 'msmarco-full@4000000')" cpu 100 30
timing 8m_exact msmarco-full - bridge_full exact "$(lex msmarco-full)" cpu 100 30
timing 8m_hnsw msmarco-full - bridge_full "$FULL_INDEX" "$(lex msmarco-full)" - 200 30
timing 5m hotpotqa-full - bridge_hp exact "$(lex hotpotqa-full)" cpu 100 30
timing long171k trec-covid - bridge_tc exact "$(lex trec-covid)" - 49 30

# 5. New corpora, in the plan's order, encoding them first.
run fresh_lotte --dataset lotte-technology $PAIRS $FRESH
run fresh_fever --dataset fever-1m $PAIRS $FRESH
run fresh_touche --dataset touche2020 $PAIRS $FRESH
guard fit_touche $QT fit touche2020 $C --device cuda --bridge "$OUT/bridge_touche"
[ -f "$OUT/fit_touche.done" ] && timing long382k touche2020 - bridge_touche exact "$(lex touche2020)" - 48 20

[ -n "$skipped" ] && echo "SKIPPED, out of time:$skipped"
if [ -n "$failed" ]; then echo "REVIEW EXPERIMENTS FAILED:$failed $(date '+%F %T')"; exit 1; fi
echo "REVIEW EXPERIMENTS DONE $(date '+%F %T')"
