RLLM_HOME=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
source $(conda info --base)/etc/profile.d/conda.sh
num_gpu=8

cd $RLLM_HOME

conda activate retriever
python agent/search/retrieval/retrieval_server_searchr1.py --faiss_gpu > server_branpo.log &
PID=$!

conda activate rllm

ex_name=qwen2.5-7b-instruct-branpo-searchr1
model_path="path/to/sft/model"

bash agent/search/train_branch_rl_script.sh \
    --model-name $output_dir \
    --train-data train \
    --ex-name $ex_name \
    --gpu $num_gpu \
    --max-resp-len $((1024*3)) \
    --max-turns 4 \
    --start-step 100 \
    --total-steps 300 \
    --n 4 \
    --lr 1e-6 > log_train_$ex_name.log 2>&1

output_dir=$RLLM_HOME/checkpoints/rllm-searchr1/$ex_name
file=$output_dir/latest_checkpointed_iteration.txt
step=$(cat "$file")
echo "Merge at step $step"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $output_dir/global_step_$step/actor \
    --target_dir $output_dir/global_step_$step/merged

