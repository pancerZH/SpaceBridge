# Restartable steps for long experiment drivers. After any crash, even a power loss, rerun the driver: finished steps
# are skipped, and an unfinished one continues from its last saved work.
#
#     OUT=outputs/phase1/scale; . scripts/steps.sh
#     bench ms1m_main --dataset msmarco-1m --tier 1 3 --rows 0a 0b 8b --K 8192 --seed 0 1 2
#     step ms_curve uv run python -u scripts/some_study.py "$OUT/ms_curve.json"
#
# Each step logs to $OUT/<name>.log, and $OUT/progress.log gets one synced line per event (cell, encoding part, rerank
# chunk), so `tail $OUT/progress.log` shows where a run is, or was when the machine went down.
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPACEBRIDGE_PROGRESS="$OUT/progress.log"

step() {  # name, then the command
  local name=$1
  shift
  if [ -f "$OUT/$name.done" ]; then
    echo "skip $name (done)"
    return 0
  fi
  echo "$(date '+%F %T') step $name: start" | tee -a "$SPACEBRIDGE_PROGRESS"
  "$@" >> "$OUT/$name.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch "$OUT/$name.done" && sync "$OUT/$name.done"
  echo "$(date '+%F %T') step $name: exit $rc" | tee -a "$SPACEBRIDGE_PROGRESS"
  return $rc
}

bench() {  # name, then spacebridge-bench arguments; results go to $OUT/<name>.jsonl
  local name=$1
  shift
  step "$name" uv run python -u -m spacebridge_bench.run "$@" --out "$OUT/$name.jsonl"
}
