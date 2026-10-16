#!/usr/bin/env bash
# Synthesize sa_top with Yosys onto the SkyWater SKY130 HD standard cells
# (typical corner, 25 C, 1.8 V) and report area plus an ABC timing estimate.
#
# Usage: synth/run_synth.sh ROWS COLS [PIPE_MAC] [TARGET_PS]
# TARGET_PS is the delay target handed to ABC's mapper and gate sizer
# (timing-driven mapping). 0 means area-oriented mapping.
# Needs synth/lib/sky130_fd_sc_hd__tt_025C_1v80.lib (synth/get_lib.sh fetches it).
set -euo pipefail
cd "$(dirname "$0")"
ROWS=${1:-4}
COLS=${2:-4}
PIPE=${3:-0}
TARGET=${4:-0}
LIB=lib/sky130_fd_sc_hd__tt_025C_1v80.lib
TAG=${ROWS}x${COLS}$([ "$PIPE" = 1 ] && echo p || true)_t${TARGET}
OUT=out/$TAG
mkdir -p "$OUT"

# ABC script. Yosys hands ABC only the combinational logic between
# registers, so the reported delay (stime -p) is register-to-register logic
# delay: no clock-to-Q, setup or wire parasitics.
#   TARGET=0  area-oriented: aggressive AIG rewriting, then area mapping.
#             This collapses fast adder structures into ripple carry chains.
#   TARGET>0  delay-oriented: choice-based mapping with a delay target, then
#             gate sizing toward the same target.
if [ "$TARGET" = 0 ]; then
cat > "$OUT/abc.script" <<ABC
strash
&get -n; &fraig -x; &put
dc2; strash
&get -n; &dch -f; &nf; &put
buffer; upsize; dnsize
stime -p
ABC
else
cat > "$OUT/abc.script" <<ABC
strash
dch
map -D $TARGET
buffer
upsize -D $TARGET
dnsize -D $TARGET
stime -p
ABC
fi

yosys -q -l "$OUT/yosys.log" -p "
  read_verilog -sv ../rtl/pe.sv ../rtl/systolic_array.sv ../rtl/sa_top.sv
  chparam -set ROWS $ROWS -set COLS $COLS -set PIPE_MAC $PIPE sa_top
  synth -top sa_top -flatten -noabc -noalumacc
  dfflibmap -liberty $LIB
  abc -liberty $LIB -script $OUT/abc.script
  opt_clean
  tee -o $OUT/stat.txt stat -liberty $LIB
  write_verilog -noattr $OUT/sa_top_netlist.v
"
echo "== $TAG =="
grep -E "Chip area|Number of cells" "$OUT/stat.txt" | head -3
grep -E "Delay =" "$OUT/yosys.log" | tail -1
