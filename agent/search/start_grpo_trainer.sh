set -x

TEMP=$(getopt -o u:p:h --long model-name:,train-data:,ex-name:,gpu:,max-resp-len:,max-turns:,start-step:,total-steps:,lr:,help -n 'train_rl_script.sh' -- "$@")
eval set -- "$TEMP"

while true; do
  case "$1" in
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --ex-name) EX_NAME="$2"; shift 2 ;;
    --gpu) NUM_GPU="$2"; shift 2 ;;
    --max-resp-len) MAX_RES_LEN="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --start-step) START_STEP="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Error!"; exit 1 ;;
  esac
done


export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_USE_FLASHINFER_SAMPLER=0

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH="${RLLM_DIR}:${PYTHONPATH}"

python3 -m agent.search.train_searchr1_agent \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0\
    algorithm.rollout_correction.rollout_is_batch_normalize=True \
    algorithm.rollout_correction.bypass_mode=True \
    data.train_data=$TRAIN_DATA \
    data.test_data=test-asearcher \
    data.train_batch_size=256 \
    data.val_batch_size=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RES_LEN \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    rllm.mask_truncated_samples=False \
    rllm.agent.max_steps=$MAX_TURNS \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb', 'console'] \
    trainer.project_name='rllm-searchr1' \
    trainer.experiment_name=$EX_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.start_step=$START_STEP \
    trainer.total_steps=$TOTAL_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 