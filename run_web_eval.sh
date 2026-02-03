RLLM_HOME=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
source $(conda info --base)/etc/profile.d/conda.sh

cd $RLLM_HOME
conda activate rllm
num_gpu=4
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve --config agent/search/vllm_server_qwen30b.yaml \
                                        --tensor-parallel-size 2 \
                                        --data-parallel-size $(($num_gpu/2)) \
                                        > vllm_server_eval.log &
PID=$!


export CUDA_VISIBLE_DEVICES=0,1,2,3

export OPENAI_API_KEY="api_key"
export OPENAI_BASE_URL="http://localhost:8001/v1"
export SUMMARY_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export JINA_API_KEY=''

python agent/search/web/test_searchr1_agent.py \
     --run_name $ex_name --model_path $model_path \
     --test_data 'test-gaia' \
     --pass_at_k 1 \
     --agent_max_steps 32 \
     --temperature 0.7 \
     --top_p 0.8 \
     --max_model_len $((1024*16)) \
     --tp_size 2 \
     --dp_size $(($num_gpu/2))

kill $PID