#!/bin/zsh
# 用冻结配置运行 gfsim 并打印 Total Cycles。
#   ./run_perf.sh <elf> <det|random300> [RWDB] [WAY]
# 模型固定为 SuperScalarModel-pr1177 @ 18b24f6ca（见 archive 根 README 的缺口说明）。
set -e
A=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177}
elf=$1; cfg=$2; rw=${3:-}; way=${4:-}
COMMON=(-s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true
  -s bctrl.vec_cell_sched_enable=true -s core.read_bandwidth_limit=128
  -s core.write_bandwidth_limit=128 -s core.sharetreg_size_bytes=262144
  -s core.sharetreg_ssb_count=64)
if [ "$cfg" = "random300" ]; then
  X=(-s core.soc_random=true -s core.mean_read_latency=300 -s core.mean_write_latency=300)
elif [ "$cfg" = "det" ]; then
  X=(-s core.soc_random=false -s core.soc_enable=false)
else
  echo "cfg must be det|random300" >&2; exit 2
fi
extra=()
[ -n "$rw" ] && extra+=(-s sl2.RWDB_ENTRY_COUNT=$rw)
[ -n "$way" ] && extra+=(-s sl2.TAG_WAY_COUNT=$way)
cd "$MODEL"
./bin/gfsim --test-finisher 1 -f "$elf" "${COMMON[@]}" "${X[@]}" "${extra[@]}" 2>&1 \
  | tee /tmp/hash_archive_run.log | grep -E "^Total Cycles"
