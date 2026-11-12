#!/bin/zsh
# 用冻结配置运行 gfsim 并打印 Total Cycles。
#   ./run_perf.sh <elf> <det|random300> [RWDB] [WAY]
# 模型固定为 SuperScalarModel-pr1177 @ 18b24f6ca（见 archive 根 README 的缺口说明）。
#
# 必填环境变量（归档不保存个人路径）：
#   MODEL  SuperScalarModel-pr1177 工作树根目录；
#          其 bin/gfsim 的 sha256 必须是 12066cbdc29466d95cce1229129ff5fa03edbdd22880fbad57d55caec87633d8
# 可选：
#   LOG    运行日志路径（默认写 $TMPDIR/hash_archive_run.XXXXXX）
set -e
A=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL:?set MODEL to the SuperScalarModel-pr1177 worktree root}"
[ -x "$MODEL/bin/gfsim" ] || { echo "missing $MODEL/bin/gfsim" >&2; exit 2; }
elf=$1; cfg=$2; rw=${3:-}; way=${4:-}
[ -f "$elf" ] || { echo "missing ELF: $elf" >&2; exit 2; }
LOG=${LOG:-$(mktemp "${TMPDIR:-/tmp}/hash_archive_run.XXXXXX")}
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
  | tee "$LOG" | grep -E "^Total Cycles"
echo "log: $LOG" >&2
