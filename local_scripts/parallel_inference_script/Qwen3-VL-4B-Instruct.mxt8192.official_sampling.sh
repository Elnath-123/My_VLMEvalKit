

# export LMUData=<TSV_Filefolder>
export http_proxy=""
export https_proxy=""

export OPENAI_API_KEY="sk-placeholder"
export OPENAI_API_BASE="http://10.223.16.13:8000/v1/chat/completions"
export LMUData="/root/paddlejob/workspace/env_run/output/rqli/datasets/LMUData"
MODEL=Qwen3-VL-4B-Instruct.mxt8192.official_sampling

WORK_BASE=/root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit

DATASETS=(
  # "MathVista_MINI"
  # "MathVision_MINI"
  # "MMBench_DEV_EN"
  # "MMStar"
  # "LogicVista"
  # "BLINK"
  # "HallusionBench"
  # "OCRBench"
  "MathVerse_MINI"
)

SUFFIX="results"

# 8 张卡数据并行：每张卡部署一个模型实例，数据集按 stride=8 自动切片，
# run.py 顶部会根据 LOCAL_RANK 自动设置每个进程的 CUDA_VISIBLE_DEVICES。
# 注意：vLLM 在 torchrun 多进程模式下，每个进程只看到自己的 1 张卡，
#       tp_size=1，彼此独立推理，最终由 RANK 0 汇总结果。
NGPU=8

SCRIPT_NAME="$(basename "$0" .sh)"
TIMESTAMP="$(date +%Y%m%d%H%M)"
LOG_DIR="${WORK_BASE}/local_scripts/eval_log/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

for DATA in "${DATASETS[@]}"; do

  WORK_DIR="${WORK_BASE}/${SUFFIX}"
  LOG_FILE="${LOG_DIR}/${DATA}.log"
  echo "[$(date '+%H:%M:%S')] START  ${DATA}  → ${LOG_FILE}"
  torchrun \
    --nproc-per-node=${NGPU} \
    --master-port=29500 \
    /root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit/run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --judge "Qwen3-235B-A22B-Instruct-2507" \
    --work-dir "${WORK_DIR}" \
    --reuse \
    > "${LOG_FILE}" 2>&1
  echo "[$(date '+%H:%M:%S')] DONE   ${DATA}  (exit=$?)"

done

echo "===== 所有任务完成  日志目录:${LOG_DIR} ====="
