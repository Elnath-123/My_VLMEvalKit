#!/usr/bin/env bash
set -euo pipefail
export http_proxy=""
export https_proxy=""
# ── 基本配置 ────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="sk-placeholder"
export OPENAI_API_BASE="http://10.223.16.13:8000/v1/chat/completions"
export LMUData="/root/paddlejob/workspace/env_run/output/rqli/datasets/LMUData"

MODEL=Qwen3-VL-4B-Instruct-lightSFT.ep1.qwen35_nothink_distill_data.mxt8192.official_sampling
WORK_BASE=/root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit
SUFFIX="results"

NUM_GPUS=8          # 可用 GPU 数量
POLL_INTERVAL=30    # 轮询间隔（秒）

DATASETS=(
  "MathVista_MINI"
  "MathVision_MINI"
  "MathVerse_MINI"
  "MMBench_DEV_EN"
  "MMStar"
  "LogicVista"
  "BLINK"
  "HallusionBench"
  # "TableVQABench"
  "OCRBench"
)

SCRIPT_NAME="$(basename "$0" .sh)"
TIMESTAMP="$(date +%Y%m%d%H%M)"
LOG_DIR="${WORK_BASE}/local_scripts/eval_log/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# ── 状态追踪 ─────────────────────────────────────────────────────────────────
# pid_to_cuda[pid]    = cuda_id
# pid_to_dataset[pid] = dataset_name
# cuda_status[cuda_id] = "idle" | dataset_name
declare -A pid_to_cuda
declare -A pid_to_dataset
declare -A cuda_status

# 初始化 cuda_status
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  cuda_status[$gpu]="idle"
done

# 待调度队列（索引）
next_idx=0
total=${#DATASETS[@]}

# 已完成 / 已失败计数
done_count=0
fail_count=0

# ── 启动单个任务 ──────────────────────────────────────────────────────────────
launch_job() {
  local cuda_id=$1
  local dataset=$2
  local log_file="${LOG_DIR}/${dataset}_cuda_${cuda_id}.log"
  local work_dir="${WORK_BASE}/${SUFFIX}"

  VLLM_PORT=$((29500 + cuda_id)) \
  CUDA_VISIBLE_DEVICES="${cuda_id}" \
  python "${WORK_BASE}/run.py" \
    --data  "${dataset}" \
    --model "${MODEL}" \
    --judge "Qwen3-235B-A22B-Instruct-2507" \
    --work-dir "${work_dir}" \
    --verbose \
    > "${log_file}" 2>&1 &

  local pid=$!
  pid_to_cuda[$pid]=$cuda_id
  pid_to_dataset[$pid]=$dataset
  cuda_status[$cuda_id]=$dataset
  echo "[$(date '+%H:%M:%S')] LAUNCH  cuda:${cuda_id}  ${dataset}  (pid=${pid})"
}

# ── 打印实时状态面板 ───────────────────────────────────────────────────────────
print_status() {
  echo ""
  echo "┌──────────────────────────────────────────────────────┐"
  printf  "│  %-52s│\n" "状态面板  $(date '+%H:%M:%S')   完成:${done_count}/${total}  失败:${fail_count}"
  echo "├──────┬───────────────────────────────────────────────┤"
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    local status="${cuda_status[$gpu]}"
    if [[ "$status" == "idle" ]]; then
      printf "│ GPU%-2d │ %-45s│\n" "$gpu" "(idle)"
    else
      printf "│ GPU%-2d │ %-45s│\n" "$gpu" "$status"
    fi
  done
  echo "├──────┴───────────────────────────────────────────────┤"
  # 显示待调度队列
  local remaining=$(( total - next_idx ))
  printf  "│  %-52s│\n" "待调度队列 (${remaining} 个): ${DATASETS[*]:$next_idx}"
  echo "└──────────────────────────────────────────────────────┘"
  echo ""
}

# ── 阶段1：启动前 min(NUM_GPUS, total) 个任务 ─────────────────────────────────
echo "===== 调度器启动  总任务=${total}  GPU数=${NUM_GPUS} ====="
initial=$(( total < NUM_GPUS ? total : NUM_GPUS ))
for ((i=0; i<initial; i++)); do
  launch_job "$i" "${DATASETS[$i]}"
done
next_idx=$initial
print_status

# ── 阶段2：轮询，动态调度剩余任务 ─────────────────────────────────────────────
while [[ ${#pid_to_cuda[@]} -gt 0 ]]; do
  sleep "$POLL_INTERVAL"

  # 扫描所有在跑的 pid
  for pid in "${!pid_to_cuda[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # 进程已结束，获取退出码
      wait "$pid" 2>/dev/null; exit_code=$?
      cuda_id=${pid_to_cuda[$pid]}
      dataset=${pid_to_dataset[$pid]}

      if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] DONE    cuda:${cuda_id}  ${dataset}  (exit=0)"
        (( done_count++ )) || true
      else
        echo "[$(date '+%H:%M:%S')] FAILED  cuda:${cuda_id}  ${dataset}  (exit=${exit_code}) → 跳过"
        (( fail_count++ )) || true
      fi

      # 从映射表中移除
      unset pid_to_cuda[$pid]
      unset pid_to_dataset[$pid]
      cuda_status[$cuda_id]="idle"

      # 如果还有待调度的数据集，立即补上
      if [[ $next_idx -lt $total ]]; then
        launch_job "$cuda_id" "${DATASETS[$next_idx]}"
        (( next_idx++ ))
      fi
    fi
  done

  print_status
done

# ── 阶段3：收尾 ───────────────────────────────────────────────────────────────
echo "===== 所有任务完成  成功:${done_count}  失败:${fail_count}  日志目录:${LOG_DIR} ====="
