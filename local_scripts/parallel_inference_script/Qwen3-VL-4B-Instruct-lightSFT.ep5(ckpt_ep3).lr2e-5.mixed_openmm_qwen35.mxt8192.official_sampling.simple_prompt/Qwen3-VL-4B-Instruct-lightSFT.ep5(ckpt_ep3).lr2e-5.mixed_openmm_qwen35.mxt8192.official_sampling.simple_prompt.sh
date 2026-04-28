#!/bin/bash

# export LMUData=<TSV_Filefolder>
export http_proxy=""
export https_proxy=""

export OPENAI_API_KEY="sk-placeholder"
export OPENAI_API_BASE="http://10.223.16.13:8000/v1/chat/completions"
export LMUData="/root/paddlejob/workspace/env_run/output/rqli/datasets/LMUData"

WORK_BASE=/root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="$(basename "$0" .sh)"
CONFIG_JSON="${SCRIPT_DIR}/${SCRIPT_NAME}.json"
MODEL="${SCRIPT_NAME}"

export MODEL_CONFIG_JSON="${CONFIG_JSON}"

DATASETS=(
  "MathVista_MINI"
  "MathVision_MINI"
  "MMBench_DEV_EN"
  "MMStar"
  "LogicVista"
  "BLINK"
  "HallusionBench"
  "OCRBench"
  "MathVerse_MINI"
)

# 8 张卡数据并行
NGPU=8

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/run_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

for DATA in "${DATASETS[@]}"; do

  LOG_FILE="${LOG_DIR}/${DATA}.log"
  echo "[$(date '+%H:%M:%S')] START  ${DATA}  → ${LOG_FILE}"

  torchrun \
    --nproc-per-node=${NGPU} \
    --master-port=29500 \
    "${WORK_BASE}/run_custom.py" \
    --data "${DATA}" \
    --model "${MODEL}" \
    --judge "Qwen3-235B-A22B-Instruct-2507-2" \
    --work-dir "${WORK_BASE}/results" \
    --reuse \
    > "${LOG_FILE}" 2>&1

  echo "[$(date '+%H:%M:%S')] DONE   ${DATA}  (exit=$?)"

  # 提取评测结果
  if grep -q "Evaluation Results:" "${LOG_FILE}" 2>/dev/null; then
    grep -A 50 "Evaluation Results:" "${LOG_FILE}" | tail -n +2 | sed '/^\[/d; /^$/d' \
      > "${LOG_DIR}/${DATA}_eval_results.txt"
  fi

done

echo "===== 所有任务完成 ====="
echo "===== 启动占卡 ====="
bash /root/paddlejob/workspace/env_run/output/rqli/utils/occupy_large.sh