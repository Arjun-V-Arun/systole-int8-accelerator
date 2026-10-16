#!/usr/bin/env bash
# Prove the RTL assertions in rtl/sa_top.sv with Yosys SAT-based k-induction.
#
# What is proven: for every input sequence starting from reset, all 11
# assertions hold on every cycle (unbounded proof, not just a bounded check).
# The datapath (multipliers, accumulators) is part of the model, so the
# "accumulators are zero after clear" property covers real arithmetic state.
#
# Usage: formal/run_formal.sh            # default configurations
#        formal/run_formal.sh 3 3 1      # one configuration: ROWS COLS PIPE_MAC
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

run() {
  local R=$1 C=$2 P=$3
  local log=logs/formal_${R}x${C}_p${P}.log
  yosys -q -l "$log" -p "
    read_verilog -sv -formal -DFORMAL -DSA_CHECKS_ON ../rtl/pe.sv ../rtl/systolic_array.sv ../rtl/sa_top.sv
    hierarchy -top sa_top -chparam ROWS $R -chparam COLS $C -chparam PIPE_MAC $P
    prep -top sa_top -flatten
    async2sync
    dffunmap
    sat -tempinduct -prove-asserts -set-init-zero -maxsteps 40
  " > /dev/null 2>&1
  local n_assert depth
  n_assert=$(grep "Import proof for assert" "$log" | sort -u | wc -l)
  depth=$(grep -c "Induction step failed" "$log")
  if grep -q "Induction step proven: SUCCESS" "$log"; then
    echo "${R}x${C} PIPE_MAC=$P: PROVEN  ($n_assert assertions, induction length $((depth + 1)))"
    return 0
  else
    echo "${R}x${C} PIPE_MAC=$P: FAILED  (see $log)"
    return 1
  fi
}

if [ $# -eq 3 ]; then
  run "$1" "$2" "$3"; exit $?
fi
ok=0
for cfg in "1 1 0" "1 1 1" "2 2 0" "2 2 1" "3 3 0" "3 3 1" "2 4 0" "4 4 0" "4 4 1"; do
  run $cfg || ok=1
done
exit $ok
