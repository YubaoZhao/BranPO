RLLM_HOME=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
source $(conda info --base)/etc/profile.d/conda.sh
num_gpu=8

cd $RLLM_HOME
conda activate retriever
python agent/search/retrieval/retrieval_server_asearcher.py --faiss_gpu > server_eval.log &
PID=$!


export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

ex_name=qwen2.5-7b-instruct-branpo
ckpt_dir=$RLLM_HOME/checkpoints/rllm-searchr1/$ex_name
step=80
# echo "Merge at step $step"
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir $ckpt_dir/global_step_$step/actor \
#     --target_dir $ckpt_dir/global_step_$step/merged
# model_path=$RLLM_HOME/checkpoints/rllm-searchr1/$ex_name/global_step_$step/merged

model_path=""

python agent/search/test_searchr1_agent.py \
     --run_name $ex_name --model_path $model_path \
     --test_data 'test-asearcher' \
     --pass_at_k 1 \
     --agent_max_steps 16 \
     --temperature 0.7 \
     --top_p 0.8 \
     --max_model_len $((1024*16)) \
     --tp_size 2 \
     --dp_size $(($num_gpu/2))

kill $PID
