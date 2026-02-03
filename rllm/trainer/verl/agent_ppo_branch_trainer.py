import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
#     compute_advantage,
    compute_response_mask,
    apply_kl_penalty
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
# from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from rllm.engine.agent_branch_engine import AsyncAgentExecutionEngine
from collections import defaultdict

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        masks = response_mask.sum(dim=-1).numpy()
        bsz = scores.shape[0]
        for i in range(bsz):
            if masks[i] > 0:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if masks[i] == 0:
                scores[i] = 0
                continue
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def compute_advantage(
    data: DataProto,
    adv_estimator=None,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Call compute_grpo_outcome_advantage with parameters matching its definition
    base_advantages, base_returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_scores_base"],
        # token_level_rewards=data.batch["token_level_scores"],
        response_mask=data.batch["base_response_mask"],
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    branch_advantages, branch_return = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_scores"],
        response_mask=data.batch["branch_response_mask"],
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
    )
    with torch.no_grad():
        preferences = data.batch["token_level_preferences"].sum(dim=-1)
        branch_pref_advantages = preferences.unsqueeze(-1) * data.batch["branch_first_response_mask"]


        branch_combined_advantages = torch.where(
            (branch_pref_advantages < 0) & (branch_advantages > 0),
            torch.zeros_like(branch_advantages),
            branch_advantages
        )

        advantage = base_advantages + branch_combined_advantages
    data.batch["advantages"] = advantage
    data.batch["returns"] = advantage
    return data

class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        assert self.agent_class is not None and self.env_class is not None, "Agent and environment classes must be provided"
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        start_step = self.config.trainer.get("start_step", 0)
        self.global_steps = 0
        total_max_steps = self.config.trainer.get("total_steps", 0)
        if total_max_steps > 0:
            self.total_training_steps = total_max_steps
            print(f"Total max steps: ", self.total_training_steps)
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1
        from itertools import groupby
        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                if self.global_steps <= start_step:
                    self.global_steps += 1
                    continue
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                    base_ids = list(final_gen_batch_output.non_tensor_batch['base_uid'])
                    repeat_times = [len(list(group)) for key, group in groupby(base_ids)]
                    batch = batch.sample_level_repeat(repeat_times)

                    batch = batch.union(final_gen_batch_output)
                    metrics.update(generate_metrics)
                    
                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories

                        reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            # full_branch_score = batch.non_tensor_batch['original_rewards']
                            # full_sequence_score = []
                            # for _, group in groupby(enumerate(base_ids), key=lambda x: x[1]):
                            #     first_index = next(group)[0] 
                            #     full_sequence_score.append(full_branch_score[first_index])
                            
                            # metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            # metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            # metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            # Round down to the nearest multiple of world size
                            num_trainer_replicas = self.actor_rollout_wg.world_size
                            max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                            if not max_batch_size:
                                # give up, you got everything either all wrong or right.
                                continue

                            size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if self.config.rllm.mask_truncated_samples:
                            mask = batch.batch["attention_mask"][:, -1] == 1
                            batch = batch[~mask]

                        batch = self._pad_dataproto_to_world_size(batch=batch)
                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                            apply_rollout_correction(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)



                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None and self.global_steps % self.config.trainer.save_freq != 0:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.global_steps % self.config.trainer.save_freq != 0:
                        self._save_checkpoint()
                    return
                self.global_steps += 1

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            data_sources = []
            for item in test_data['extra_info']:
                data_sources.append(item.get('data_source', 'unknown'))
            test_data['data_source'] = np.array(data_sources)
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info, save_dir_name="val_chat_completions", num_branches=1)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict


    def generate_agent_trajectory(self, timing_raw=None, meta_info=None, save_dir_name=None, num_branches=-1):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token", num_branches=num_branches)
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories, save_dir_name)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                steps.append(trajectory)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict], save_dir_name: str=None):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.
        Now flattens all branches into the batch and separates base/branch masks and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length


        all_initial_tokens_list = []
        all_response_tokens_list = []
        
        # New masks lists
        all_base_response_masks_list = []
        all_branch_response_masks_list = []
        all_branch_first_response_masks_list = []
        all_masks_list = []
        # New reward lists
        all_base_rewards_value = []   # Stores scalar value temporarily
        all_branch_rewards_value = [] # Stores scalar value temporarily
        all_preferences_value = []
        # For tracking lengths to place rewards correctly later
        base_response_lengths = []
        branch_response_lengths = []

        all_logprobs_list = []
        traj_metrics = []
        metrics = {}
        
        # List to store data for jsonl logging
        jsonl_logs = []
        base_ids = []
        repeate_times = []
        original_rewards = []
    
        for base_id, branch_traj in enumerate(trajectories):
            base_data = branch_traj['base']
            base_prompt = base_data["prompt_tokens"]
            base_response = base_data["response_tokens"]
            base_mask = base_data["response_masks"] # Assuming 1s for valid tokens
            branches = branch_traj['branches']

            # Key Logic 2: Base reward is the max of all branches (Value Estimation for the base state)
            base_reward_scalar = np.mean(branch_traj['rewards'])
            branch_rewards = branch_traj['rewards']

            for i, branch in enumerate(branches):
                base_ids.append(base_id)
                branch_response = branch["response_tokens"]
                branch_mask = branch["response_masks"]
                branch_first_mask = branch['first_response_masks']
                branch_reward_scalar = branch_rewards[i]

                # --- 1. Construct Combined Response ---
                # Response = Base Response + Branch Response
                combined_response = torch.cat([base_response, branch_response], dim=0)
                
                # --- 2. Construct Separated Masks ---
                # Base Mask: 1s in base part, 0s in branch part
                # Branch Mask: 0s in base part, 1s in branch part
                
                # Create padding of zeros for the other section
                zeros_for_branch = torch.zeros_like(branch_response)
                zeros_for_base = torch.zeros_like(base_response)
            
                combined_branch_mask = torch.cat([zeros_for_base, branch_mask], dim=0)
                if i == 0:
                    combined_base_mask = torch.cat([base_mask, zeros_for_branch], dim=0)
                    traj_response_mask = torch.cat([base_mask, branch_mask])
                else:
                    combined_base_mask = torch.cat([zeros_for_base, zeros_for_branch], dim=0)
                    traj_response_mask = torch.cat([zeros_for_base, branch_mask])
                
                # --- 3. Collect Data ---
                all_initial_tokens_list.append(base_prompt)
                all_response_tokens_list.append(combined_response)
                
                all_base_response_masks_list.append(combined_base_mask)
                all_branch_response_masks_list.append(combined_branch_mask)
                all_branch_first_response_masks_list.append(torch.cat([zeros_for_base, branch_first_mask], dim=0))
                all_masks_list.append(traj_response_mask)
                # Store scalar rewards and lengths to build tensors later
                all_base_rewards_value.append(base_reward_scalar)
                all_branch_rewards_value.append(branch_reward_scalar)
                all_preferences_value.append(branch['preference'])

                base_response_lengths.append(len(base_response))
                branch_response_lengths.append(len(branch_response))

                # Handle Logprobs: Concatenate base logprobs + branch logprobs
                # Assuming 'rollout_log_probs' exists in both or needs handling
                base_logprobs = base_data.get("rollout_log_probs", torch.zeros(len(base_response)))
                branch_logprobs = branch.get("rollout_log_probs", torch.zeros(len(branch_response)))
                combined_logprobs = torch.cat([base_logprobs, branch_logprobs], dim=0)
                all_logprobs_list.append(combined_logprobs)

                # Metrics and Logs
                if "metrics" in branch_traj:
                    traj_metrics.append(branch_traj["metrics"])
                
                # Prepare log entry
                if 'chat_completions' in branch:
                    log_entry = {"messages": branch['chat_completions'], "task": branch_traj.get('task', ''), 'branch': i}
                    jsonl_logs.append(log_entry)
        # --- Metrics Processing ---
        if traj_metrics:
            # Flatten traj_metrics into a dict of lists
            traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
            branch_metrics = ["num_branches", "efficient_branch", "reward_branch"]
            for k, v_list in traj_metrics.items():
                v_list = [v for v in v_list if v is not None and v >= 0]
                if not v_list: continue
                v_list = np.array(v_list)
                if k in branch_metrics:
                    metrics.update({
                        f"traj/{k}_mean": v_list.mean(),
                    })
                else:
                    metrics.update({
                        f"traj/{k}_mean": v_list.mean(),
                        f"traj/{k}_min": v_list.min(),
                        f"traj/{k}_max": v_list.max(),
                    })

        # --- Save JSONL Logs ---
        if save_dir_name is None:
            save_dir_name = "chat_completions"
        save_dir = os.path.join(self.config.trainer.default_local_dir, save_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        open_mode = 'a+' if save_dir_name != "chat_completions" else 'w'
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), open_mode) as f:
            for log in jsonl_logs:
                f.write(json.dumps(log) + "\n")

        # --- Padding & Batch Construction ---
        
        # 1. Pad Prompts (Left Pad)
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]
        prompts_batch = prompts_batch.to(torch.long)

        # 2. Pad Responses (Right Pad)
        # Note: These responses are now Base + Branch
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]
        response_batch = response_batch.to(torch.long)

        # 3. Pad Logprobs
        logprobs_batch = torch.nn.utils.rnn.pad_sequence(all_logprobs_list, batch_first=True, padding_value=0.0)
        logprobs_batch = pad_sequence_to_length(logprobs_batch, max_response_length, 0.0, left_pad=False)
        logprobs_batch = logprobs_batch[:, :max_response_length]

        # 4. Pad Masks (Base & Branch Separately)
        # Base Mask
        base_mask_batch = torch.nn.utils.rnn.pad_sequence(all_base_response_masks_list, batch_first=True, padding_value=0)
        base_mask_batch = pad_sequence_to_length(base_mask_batch, max_response_length, 0, left_pad=False)
        base_mask_batch = base_mask_batch[:, :max_response_length]
        
        # Branch Mask
        branch_mask_batch = torch.nn.utils.rnn.pad_sequence(all_branch_response_masks_list, batch_first=True, padding_value=0)
        branch_mask_batch = pad_sequence_to_length(branch_mask_batch, max_response_length, 0, left_pad=False)
        branch_mask_batch = branch_mask_batch[:, :max_response_length]

        # Branch First Mask
        branch_first_mask_batch = torch.nn.utils.rnn.pad_sequence(all_branch_first_response_masks_list, batch_first=True, padding_value=0)
        branch_first_mask_batch = pad_sequence_to_length(branch_first_mask_batch, max_response_length, 0, left_pad=False)
        branch_first_mask_batch = branch_first_mask_batch[:, :max_response_length]

        # Combine for total mask (optional, but good for attention masking or standard loss)
        total_loss_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        total_loss_mask = pad_sequence_to_length(total_loss_mask, max_response_length, 0, left_pad=False)
        total_loss_mask = total_loss_mask[:, :max_response_length]

        # 5. Build Token-Level Rewards
        # We need two separate reward tensors.
        
        # Tensor A: Base Reward (placed at the last token of the base response)
        score_batch_base = torch.zeros_like(response_batch, dtype=torch.float32)
        # Tensor B: Branch Reward (placed at the last token of the branch response)
        score_batch_branch = torch.zeros_like(response_batch, dtype=torch.float32)
        pref_batch_branch = torch.zeros_like(response_batch, dtype=torch.float32)
        for i in range(len(all_response_tokens_list)):
            # Index for Base Reward: Length of Base - 1
            idx_base = base_response_lengths[i] - 1
            if 0 <= idx_base < max_response_length:
                score_batch_base[i, idx_base] = all_base_rewards_value[i]
            
            # Index for Branch Reward: Length of Base + Length of Branch - 1
            # Note: The combined response length is len(base) + len(branch)
            total_len = base_response_lengths[i] + branch_response_lengths[i]
            idx_branch = total_len - 1
            if 0 <= idx_branch < max_response_length:
                score_batch_branch[i, idx_branch] = all_branch_rewards_value[i]
                pref_batch_branch[i, idx_branch] = all_preferences_value[i]

        # 6. Standard Input IDs and Attention Mask Construction
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))
        
        response_real_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask_attn = resp_pos < response_real_lengths.unsqueeze(1)
        
        attention_mask = torch.cat([prompt_mask, response_mask_attn], dim=1).long()
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # 7. Final Dictionary Construction
        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "rollout_log_probs": logprobs_batch,
            
            "response_mask": total_loss_mask,          # General mask (union of base and branch)
            "base_response_mask": base_mask_batch,     # Specific mask for base part
            "branch_response_mask": branch_mask_batch, # Specific mask for branch part
            "branch_first_response_mask": branch_mask_batch,
            
            "token_level_scores_base": score_batch_base,     # Reward for base part
            "token_level_scores": score_batch_branch, # Reward for branch part
            "token_level_preferences": pref_batch_branch, # Reward for branch part
        }
        non_tensor_batch = {
            'base_uid': base_ids, 
            'preferences': all_response_tokens_list
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token", num_branches=2):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info, num_branches=num_branches):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch['base_uid'][idx] = -idx
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
