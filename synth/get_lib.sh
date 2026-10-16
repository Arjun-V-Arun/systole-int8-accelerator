#!/usr/bin/env bash
# Fetch the SKY130 HD typical-corner liberty file used by run_synth.sh.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lib
curl -sSL -o lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
  https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
echo "saved lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
