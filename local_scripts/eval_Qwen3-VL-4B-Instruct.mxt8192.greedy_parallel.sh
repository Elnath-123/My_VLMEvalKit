
SCRIPT_NAME="$(basename "$0" .sh)"
TIMESTAMP="$(date +%Y%m%d%H%M)"
export http_proxy=""
export https_proxy=""
# export no_proxy="baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com,gz.bcebos.com,10.223.16.13"
# export LMUData=<TSV_Filefolder>

export OPENAI_API_KEY="sk-placeholder"
export OPENAI_API_BASE="http://10.223.16.13:8000/v1/chat/completions"
export LMUData="/root/paddlejob/workspace/env_run/output/rqli/datasets/LMUData"
MODEL=Qwen3-VL-4B-Instruct.mxt8192.greedy

WORK_BASE=/root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit

DATASETS=(
  "MathVista_MINI"
  "MathVision_MINI"
  "MathVerse_MINI"
  "MMBench_DEV_EN"
  "MMStar"
  "LogicVista"
  "BLINK"
  "HallusionBench"
  "TableVQABench"
  "InfoVQA_TEST"
  "OCRBench"
)

SUFFIX="results"

LOG_DIR="${WORK_BASE}/local_scripts/eval_log/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# 每张卡对应一个数据集（按索引对齐）
for i in "${!DATASETS[@]}"; do
  DATA="${DATASETS[$i]}"
  CUDA_ID="${i}"
  LOG_FILE="${LOG_DIR}/${DATA}_cuda_${CUDA_ID}.log"
  WORK_DIR="${WORK_BASE}/${SUFFIX}"

  # 每个进程分配不同的 VLLM_PORT，避免 vllm 初始化时 TCPStore 端口冲突
  VLLM_PORT=$((29500 + CUDA_ID)) \
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" python /root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit/run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --judge "Qwen3-235B-A22B-Instruct-2507" \
    --work-dir "${WORK_DIR}" \
    --verbose \
    > "${LOG_FILE}" 2>&1 &

  echo "Launched: ${DATA} on cuda:${CUDA_ID}, log -> ${LOG_FILE}"
done

wait
echo "All datasets finished."
