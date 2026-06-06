RLLM_HOME=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
source $(conda info --base)/etc/profile.d/conda.sh
num_gpu=8

cd $RLLM_HOME

conda activate retriever
python agent/search/retrieval/retrieval_server_asearcher.py --faiss_gpu > server_grpo.log &
PID=$!

conda activate rllm

ex_name=qwen2.5-7b-instruct-grpo
# ex_name=qwen3-4b-instruct-grpo
model_path="path/to/sft/model"


# Stage 1 Training
bash agent/search/start_grpo_trainer.sh \
    --model-name $model_path \
    --train-data train-asearcher \
    --ex-name $ex_name \
    --gpu $num_gpu \
    --max-resp-len $((1024*3)) \
    --max-turns 4 \
    --start-step 0 \
    --total-steps 40 \
    --lr 5e-6 > log_train_$ex_name-step_4.log 2>&1


# Stage 2 Training
bash agent/search/start_grpo_trainer.sh \
    --model-name $model_path \
    --train-data train-asearcher \
    --ex-name $ex_name \
    --gpu $num_gpu \
    --max-resp-len $((1024*7)) \
    --max-turns 8 \
    --start-step 0 \
    --total-steps 80 \
    --lr 5e-6 > log_train_$ex_name-step_8.log 2>&1

kill $PID

