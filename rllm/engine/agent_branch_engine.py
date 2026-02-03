import asyncio
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.utils import colorful_print

logger = logging.getLogger(__name__)

import copy
import asyncio
import time
import json


class TaskAccumulator:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.task_info = {}

    def register_tasks(self, envs):
        for env in envs:
            task = json.dumps(env.task)
            if task not in self.task_info:
                self.task_info[task] = {
                    'count': 0,
                    'remaining': 0,
                    'accuracy_sum': 0.0,
                    'steps_mean': 0.0,
                    'event': None  
                }
            self.task_info[task]['count'] += 1
            self.task_info[task]['remaining'] += 1

    async def update_accuracy(self, env, accuracy, steps):
        task = json.dumps(env.task)
        async with self.lock:
            if task not in self.task_info:
                raise ValueError(f"Task {task} was not registered.")
            info = self.task_info[task]
            if accuracy >= 0.8:
                info['accuracy_sum'] += 1 
                info['steps_mean'] = ((info['accuracy_sum'] - 1) * info['steps_mean'] + steps) / info['accuracy_sum']
            info['remaining'] -= 1
            if info['event'] is None:
                info['event'] = asyncio.Event()

            if info['remaining'] == 0:
                info['event'].set()

    async def wait_for_task_accuracy(self, env):
        task = json.dumps(env.task)
        async with self.lock:
            if task not in self.task_info:
                raise ValueError(f"Task {task} was not registered.")
            info = self.task_info[task]
            count = info['count']

            if info['event'] is None:
                info['event'] = asyncio.Event()
                if count == 0:
                    info['event'].set()

        await info['event'].wait()

        async with self.lock:
            total = self.task_info[task]['accuracy_sum']
            avg_acc = total / count if count > 0 else 0.0
        return avg_acc, info['steps_mean']

class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,  # The number of active agents
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]
        self.accuracy_map = TaskAccumulator()
        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl", "vllm", "tinker"], "Currently only openai, verl and tinker are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "vllm":
            from rllm.engine.rollout.vllm_engine import vllmEngine
            self.rollout_engine = vllmEngine(
                **rollout_engine_args,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                max_num_seqs=n_parallel_agents
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "tinker":
            from rllm.engine.rollout.tinker_engine import TinkerEngine

            self.rollout_engine = TinkerEngine(
                **rollout_engine_args,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, application_id, prompt=None, prompt_ids=None, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai" or self.engine_name == "vllm":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            if prompt_ids is None:
                output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            else:
                output = await self.rollout_engine.get_model_response(request_prompt_ids=prompt_ids, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "tinker":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.accuracy_map = TaskAccumulator()
        self.accuracy_map.register_tasks(envs)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Token", num_branches=-1, **kwargs):
        """Run a single agent's trajectory asynchronously with optional branching at the last step"""
        agent = self.agents[idx]
        env = self.envs[idx]

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0
        invalid_actions = 0
        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        agent.update_tokens(prompt_tokens)

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            completion_tokens = agent.chat_completions_tokens.copy()
            
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = min(max_tokens, 1024)

            start_time = time.time()
            if self.engine_name == 'verl':
                model_output = await self.get_model_response(prompt_ids=completion_tokens, application_id=application_id, **kwargs)
            else:
                model_output = await self.get_model_response(prompt=prompt_messages, application_id=application_id, **kwargs)
            response = model_output.text
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                cur_step.reward = 0.0
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens = prompt_response_pair['completion_ids']
                assistant_msg_masks = [1] * len(assistant_msg_tokens)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                truncation_length = self.max_response_length - response_token_len
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break
            if 'format_reward' in cur_step.info and cur_step.info['format_reward'] < 0:
                invalid_actions += 1
                cur_step = agent.get_current_state()
                cur_step.reward = cur_step.info['format_reward']
                cur_step.done = True
                # colorful_print(f"Step {step_idx} of Trajectory {idx} is masked out due to invalid.", "red")  
                termination_reason = "INVALID"
                break

            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            agent.update_tokens(assistant_msg_tokens)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)
            agent.update_tokens(env_msg_tokens)
            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                response_masks = [0] * len(response_masks)
                masked_out = True
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "blue")

        if termination_reason == "INVALID":
            # Mask out the entire response for overlong trajectories if the reward is 0.
            response_masks = [0] * len(response_masks)
            masked_out = True
            colorful_print(f"Trajectory {idx} is masked out due to invalid.", "red")

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward

        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
        
        metrics_kwargs = {
            'total_time': total_time,
            'llm_time': llm_time,
            'env_time': env_time,
            'reward_time': reward_time or 0.0,
            'invalid_actions': invalid_actions,
        }
        trajectory: Trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        branch_kwargs = {
            'episode_steps': episode_steps,
            'first_branch_reward': trajectory.reward,
            'first_branch_masked_out': masked_out,
            "task_accuracy": 0.0,
            "idx":idx,
            "application_id":application_id,
            "agent":agent,
            "env":env,
            "loop":loop,
            'mode': mode,
            **metrics_kwargs,
            **kwargs
        }
        branch_results = None
        if num_branches == -1:
            step_num = len(episode_steps)
            await self.accuracy_map.update_accuracy(env, reward, step_num)
            if termination_reason not in ["ENV_TIMEOUT", "INVALID", "TRUNCATION"]:
                accuracy, mean_steps = await self.accuracy_map.wait_for_task_accuracy(env)
                branch_kwargs["task_accuracy"] = accuracy

                if accuracy > 2/4 and accuracy < 1 and reward >= 0.8 and step_num - mean_steps >= 0.1:
                    branch_results = await self._run_branching(num_branches=6,
                                               branchpoint=2,
                                            **branch_kwargs)
                elif accuracy == 1:
                    branch_results = await self._run_branching(num_branches=2,
                                               branchpoint=1,
                                            **branch_kwargs)                    
                elif accuracy < 2/4 and reward < 0.8:
                    branch_results = await self._run_branching(num_branches=3,
                                               branchpoint=min(1, len(episode_steps)),
                                            **branch_kwargs)
                    if len(branch_results["branches"]) == 1:
                        branch_results = await self._run_branching(num_branches=3,
                                               branchpoint=min(2, len(episode_steps)),
                                            **branch_kwargs)
                    if len(branch_results["branches"]) == 1:
                        branch_results = await self._run_branching(num_branches=3,
                                               branchpoint=min(3, len(episode_steps)),
                                            **branch_kwargs)
                elif accuracy <= 3/4 or reward < 0.8:
                    branch_results = await self._run_branching(num_branches=3,
                                               branchpoint=min(1, len(episode_steps)),
                                            **branch_kwargs)
                    if len(branch_results["branches"]) == 1:
                        branch_results = await self._run_branching(num_branches=3,
                                               branchpoint=min(2, len(episode_steps)),
                                            **branch_kwargs)
        if branch_results is None:
            branch_results = await self._run_branching(num_branches=1,
                                    branchpoint=min(2, len(episode_steps)),
                                **branch_kwargs)            
        # Original non-branching logic
        await loop.run_in_executor(self.executor, env.close)

        return branch_results



    async def _run_trajectory_from_point(self, agent, env, loop, application_id, start_step_idx, max_steps,
                                        response_token_len_offset, total_time, **kwargs):
        """Run trajectory from a given point until done=True
        
        Args:
            agent: Agent instance
            env: Environment instance
            loop: Event loop
            application_id: Application ID for model requests
            start_step_idx: Starting step index
            response_token_len_offset: Initial response token length
            total_time: Current total time
            **kwargs: Additional arguments for model response
            
        Returns:
            tuple: (branch_steps, reward, termination_reason, llm_time, env_time, reward_time, response_token_len)
        """
        branch_steps = []
        response_token_len = response_token_len_offset
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0
        reward_time = 0.0
        termination_reason = None
        
        for step_idx in range(start_step_idx, max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            completion_tokens = agent.chat_completions_tokens.copy()
            
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            if self.engine_name == 'verl':
                model_output = await self.get_model_response(prompt_ids=completion_tokens, application_id=application_id, **kwargs)
            else:
                model_output = await self.get_model_response(prompt=prompt_messages, application_id=application_id, **kwargs)
            response = model_output.text
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            branch_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action), 
                    timeout=(self.trajectory_timeout - total_time)
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == start_step_idx:
                    colorful_print(f"Warning: Branch completed due to: {termination_reason} before able to perform 1 complete action.\n", "red")
                cur_step.reward = 0
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens = prompt_response_pair['completion_ids']
                assistant_msg_masks = [1] * len(assistant_msg_tokens)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                    env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, 
                    contains_first_msg=False, contains_generation_msg=True
                )

            # Update response token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break
            if 'format_reward' in cur_step.info and cur_step.info['format_reward'] < 0:
                cur_step = agent.get_current_state()
                cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "INVALID"
                break

            agent.update_tokens(assistant_msg_tokens)

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            agent.update_tokens(env_msg_tokens)
            if step_idx == max_steps - 1:
                termination_reason = "MAX_STEPS"

        # Apply masking if needed
        masked_out = False

        if termination_reason != "ENV_DONE":
            masked_out = True

        # Compute final reward
        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        trajectory: Trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        return branch_steps, trajectory.reward, termination_reason, llm_time, env_time, reward_time, response_token_len, masked_out


    async def _replay_trajectory_to_point(self, agent, env, loop, episode_steps, end_step_idx):
        """Replay trajectory up to a certain step
        
        Args:
            agent: Agent instance
            env: Environment instance
            loop: Event loop
            episode_steps: List of episode steps
            end_step_idx: Index to replay up to (exclusive)
            
        Returns:
            int: Total response token length up to this point
        """
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps
        
        agent.reset()
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)
        
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages, tokenizer=self.tokenizer, parser=self.chat_parser, 
            contains_first_msg=True, contains_generation_msg=True
        )
        agent.update_tokens(prompt_tokens)
        
        response_token_len = 0
        
        for step_idx in range(end_step_idx):
            step_data = episode_steps[step_idx]
            action: Action = agent.update_from_model(step_data["response"])
            action = action.action
            
            next_observation, reward, done, info = await loop.run_in_executor(self.executor, env.step, action)
            info["max_steps"] = self.max_steps
            agent.update_from_env(observation=next_observation, reward=reward, done=done, info=info)
            
            # Update token count
            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)
            
            assistant_msg_tokens = step_data['completion_ids']
            env_msg_tokens, _ = convert_messages_to_tokens_and_masks(
                env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, 
                contains_first_msg=False, contains_generation_msg=True
            ) if env_messages else ([], [])
            
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            agent.update_tokens(assistant_msg_tokens)
            if env_messages:
                agent.update_tokens(env_msg_tokens)
        
        return response_token_len


    async def _run_branching(self, idx, application_id, agent, env, loop, episode_steps, task_accuracy, branchpoint, num_branches, invalid_actions,
                            first_branch_reward, first_branch_masked_out, total_time, llm_time, env_time, reward_time, mode, **kwargs):
        """Generate branches from the last node of the trajectory
        
        The first trajectory is considered as the first branch, so we only need to generate (num_branches - 1) additional branches.
        All branches run until done=True.
        """
        
        # Determine base and branching point
        # For a trajectory with only 1 step, base should be empty
        if len(episode_steps) <= branchpoint:
            base_steps = []
            branch_start_idx = 0
        else:
            base_steps = episode_steps[:-branchpoint]
            branch_start_idx = len(episode_steps) - branchpoint
    
        
        # Prepare base data
        if mode == "Token" and len(base_steps) > 0:
            base_prompt_tokens, base_response_tokens, base_response_masks, base_response_logprobs, _ = self.assemble_steps(base_steps)
        else:
            base_prompt_tokens = torch.tensor(episode_steps[0]["prompt_ids"].copy(), dtype=torch.long)
            base_response_tokens, base_response_masks, base_response_logprobs = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # if first_branch_masked_out:
        #     base_response_masks = torch.zeros_like(base_response_masks, dtype=torch.long)

        def prepare_branch_data(steps, masked_out, reward, messages):
            branch_prompt_ids, branch_response_tokens, branch_response_masks, branch_response_logprobs, first_response_masks, _ = self.assemble_steps(steps, return_first_mask=True)
            base_len = len(base_prompt_tokens) + len(base_response_tokens)
            base_next_env_tokens = branch_prompt_ids[base_len:]
            if len(base_next_env_tokens) > 0:
                branch_response_tokens = torch.cat([base_next_env_tokens, branch_response_tokens])
                branch_response_masks = torch.cat([torch.zeros(len(base_next_env_tokens), dtype=torch.long), branch_response_masks])
                first_response_masks = torch.cat([torch.zeros(len(base_next_env_tokens), dtype=torch.long), first_response_masks])
                branch_response_logprobs = torch.cat([torch.zeros(len(base_next_env_tokens), dtype=torch.float), branch_response_logprobs]) 

            branch_data = {
                # "prompt_tokens": branch_prompt_ids,
                "response_tokens": branch_response_tokens,
                "response_masks": branch_response_masks,
                "first_response_masks": first_response_masks,
                "rollout_log_probs": branch_response_logprobs,
                "reward": reward,
                "chat_completions": messages,
                "preference": 0,
                "steps": len(steps),
            }
            assert len(branch_response_tokens) == len(branch_response_masks)
            assert len(branch_response_tokens) == len(branch_response_logprobs)
            assert len(branch_response_tokens) == len(first_response_masks)
            return branch_data
       
        # Initialize branches list with the first trajectory (already executed)
        if mode == "Token":
            # First branch includes all steps from branch_start_idx onwards
            first_branch_steps = episode_steps[branch_start_idx:]
            branches = [prepare_branch_data(first_branch_steps, first_branch_masked_out, first_branch_reward, agent.chat_completions.copy())]
        else:
            first_branch_steps = episode_steps[branch_start_idx:]
            branches = [{
                "steps": first_branch_steps,
                "reward": first_branch_reward,
            }]
        
        # Accumulate metrics
        branch_total_llm_time = llm_time
        branch_total_env_time = env_time
        branch_total_time = total_time
        branch_total_reward_time = reward_time
        
        reward_branch = 0
        efficient_branch = 0

        # Generate additional (num_branches - 1) branches
        for branch_idx in range(1, num_branches):
            # Reset to branching point
            response_token_len = await self._replay_trajectory_to_point(agent, env, loop, episode_steps, branch_start_idx)
            
            # Run trajectory from branching point until done
            max_steps = self.max_steps
            branch_steps, reward, termination_reason, step_llm_time, step_env_time, step_reward_time, final_token_len, masked_out = await self._run_trajectory_from_point(
                agent, env, loop, application_id, branch_start_idx, max_steps, response_token_len, branch_total_time, **kwargs
            )
            
            branch_total_llm_time += step_llm_time
            branch_total_env_time += step_env_time
            branch_total_time += step_llm_time + step_env_time
            branch_total_reward_time += step_reward_time
            
            if termination_reason:
                color = "green" if reward > 0 else "yellow"
                colorful_print(
                    f"Branch {branch_idx} of Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}.\n",
                    color,
                )
            if masked_out:
                # colorful_print(f"Branch {branch_idx} of Trajectory {idx} is masked out due to overlong filter.", "red")
                continue
            reward_chosen = (first_branch_reward < 0.8 and reward >= 0.8)
            reward_rejected = (first_branch_reward >= 0.8 and reward < 0.8)
            step_rejected = (task_accuracy > 0.5 and task_accuracy < 1) and (first_branch_reward >= 0.8 and first_branch_reward == reward and len(branch_steps) > len(first_branch_steps))
            step_chosen = (task_accuracy > 0.5  and task_accuracy < 1) and (first_branch_reward >= 0.8 and first_branch_reward == reward and len(branch_steps) < len(first_branch_steps))
            if  reward_chosen or reward_rejected or step_rejected or step_chosen:
                # Prepare branch data
                if mode == "Token":
                    branch_data = prepare_branch_data(branch_steps, masked_out, reward, agent.chat_completions.copy())
                else:
                    branch_data = {
                        "steps": branch_steps,
                        "reward": reward,
                    }
                branch_data['preference'] = 1 if reward_chosen or step_chosen else -1
                branches[0]['preference'] = -branch_data['preference']
                if step_rejected or step_chosen:
                    efficient_branch += 1
                else:
                    reward_branch += 1
                branches.append(branch_data)
                break
            
        # Prepare final result
        if mode == "Token":
            result = {
                "base": {
                    "prompt_tokens": base_prompt_tokens,
                    "response_tokens": base_response_tokens,
                    "response_masks": base_response_masks,
                    "rollout_log_probs": base_response_logprobs,
                },
                "branches": branches,
                "rewards": [branch['reward'] for branch in branches],
                
                "idx": env.idx,
                "task": getattr(env, 'task', {}),
                "metrics": {
                    "steps": len(episode_steps),
                    "num_branches": len(branches),
                    "efficient_branch": efficient_branch,
                    "reward_branch": reward_branch,
                    "reward": first_branch_reward,
                    "reward_time": branch_total_reward_time,
                    "env_time": branch_total_env_time,
                    "llm_time": branch_total_llm_time,
                    "total_time": branch_total_time,
                    "invalid_actions": invalid_actions,
                },
            }
        else:
            result = {
                "base": {
                    "steps": base_steps,
                },
                "branches": branches,
                "idx": env.idx,
                "metrics": {
                    "steps": len(base_steps)+1,
                    "num_branches": len(branches),
                    "reward_time": branch_total_reward_time,
                    "env_time": branch_total_env_time,
                    "llm_time": branch_total_llm_time,
                    "total_time": branch_total_time,
                },
            }
        
        return result

    def assemble_steps(self, steps: list[dict], return_first_mask=False):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Each step_result contains:
        - steps: List of {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list}

        For training, we need to assemble the full conversation sequence where:
        - prompt_tokens: Initial prompt (first step's prompt_ids)
        - response_tokens: All subsequent conversation (completion_ids + next step's prompt_ids)
        - response_masks: Mask indicating which tokens contribute to loss (only completion_ids)
        """

        # Start with initial prompt from first step
        if len(steps) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), 1
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        first_response_masks = []
        response_logprobs = []
        is_valid_trajectory = True

        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]
            current_logprobs = step["logprobs"]

            if i == 0:
                # First step: just add completion
                response_tokens.extend(current_completion_ids)
                response_masks.extend([1] * len(current_completion_ids))  # completion contributes to loss
                first_response_masks.extend([1] * len(current_completion_ids))
                accumulated_sequence.extend(current_completion_ids)
                response_logprobs.extend(current_logprobs)
            else:
                if current_prompt_ids[: len(accumulated_sequence)] != accumulated_sequence:
                    # Find the first differing position
                    prefix = current_prompt_ids[: len(accumulated_sequence)]
                    diff_pos = None
                    for i, (expected, actual) in enumerate(zip(accumulated_sequence, prefix, strict=False)):
                        if expected != actual:
                            diff_pos = i
                            break

                    if diff_pos is not None:
                        logger.warning(f"When assemble steps, detect the trajectory not accumulative at position {diff_pos}. Expected: {accumulated_sequence[diff_pos : diff_pos + 5]}, Got: {prefix[diff_pos : diff_pos + 5]}. Setting response_masks to all 0s. This is likely due to retokenization.")
                    else:
                        logger.warning(f"When assemble steps, detect length mismatch. Expected length: {len(accumulated_sequence)}, Got length: {len(prefix)}. Setting response_masks to all 0s.")

                    is_valid_trajectory = False
                    break

                response_tokens.extend(current_prompt_ids[len(accumulated_sequence) :] + current_completion_ids)
                obs_part = len(current_prompt_ids) - len(accumulated_sequence)
                response_masks.extend([0] * obs_part + [1] * len(current_completion_ids))  # completion contributes to loss
                first_response_masks.extend([0] * obs_part + [0] * len(current_completion_ids))
                response_logprobs.extend([0.0] * obs_part + current_logprobs)
                accumulated_sequence = current_prompt_ids + current_completion_ids

        assert len(response_masks) == len(response_tokens)
        assert len(response_logprobs) == len(response_tokens)

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)
        first_response_masks = torch.tensor(first_response_masks, dtype=torch.long)
        response_logprobs = torch.tensor(response_logprobs, dtype=torch.float)

        if self.config.rllm.filter_token_mismatch:
            response_masks = response_masks * int(is_valid_trajectory)
        if not return_first_mask:
            return prompt_tokens, response_tokens, response_masks, response_logprobs, is_valid_trajectory
        else:
            return prompt_tokens, response_tokens, response_masks, response_logprobs, first_response_masks, is_valid_trajectory

    async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                application_id = str(uuid.uuid4())
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            await self.rollout_engine.wake_up()  # type: ignore

        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        async def launch_one_trajectory_task(env_idx: int):
            async with semaphore:
                try:
                    result = await self.run_agent_trajectory_with_retry(
                        idx=env_idx,
                        seed=reset_seed,
                        mode=mode,
                        **kwargs,
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise e
                return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            await self.rollout_engine.sleep()  # type: ignore

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    # res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])
        # return results
        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]

        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
