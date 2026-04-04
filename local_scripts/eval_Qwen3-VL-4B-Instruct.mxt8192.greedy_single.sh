


# export LMUData=<TSV_Filefolder>
export http_proxy=""
export https_proxy=""

export OPENAI_API_KEY="sk-placeholder"
export OPENAI_API_BASE="http://10.223.16.13:8000/v1/chat/completions"
export LMUData="/root/paddlejob/workspace/env_run/output/rqli/datasets/LMUData"
MODEL=Qwen3-VL-4B-Instruct.mxt8192.greedy

WORK_BASE=/root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit


DATASETS=(
  # "MathVista_MINI"
  "MathVision_MINI"
  # "MathVerse_MINI"
  # "MMBench_DEV_EN"
  # "MMStar"
  # "LogicVista"
  # "BLINK"
  # "HallusionBench"
  # "TableVQABench"
  # "OCRBench"
)


SUFFIX="results"

export CUDA_VISIBLE_DEVICES="0"

for DATA in "${DATASETS[@]}"; do

  WORK_DIR="${WORK_BASE}/${SUFFIX}"
  python /root/paddlejob/workspace/env_run/output/rqli/repos/VLMEvalKit/run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --judge "Qwen3-235B-A22B-Instruct-2507" \
    --work-dir "${WORK_DIR}" \
    --verbose
done