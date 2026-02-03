RLLM_HOME=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
source $(conda info --base)/etc/profile.d/conda.sh

cd $RLLM_HOME
conda activate rllm
num_gpu=8

export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve --config agent/search/vllm_server_qwen30b.yaml \
            --tensor-parallel-size 2 \
            --data-parallel-size $(($num_gpu/2)) \
            > vllm_server.log &
PID=$!

export OPENAI_API_KEY="api_key"
export OPENAI_BASE_URL="http://localhost:8001/v1"

python judge.py --names ex_name1 ex_name2 \
                --model "Qwen/Qwen3-30B-A3B-Instruct-2507"

kill $PID