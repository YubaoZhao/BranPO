import json
import queue
import re
import warnings
from typing import Any
import uuid
from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool
from rllm.environments.tools.tool_env import ToolEnvironment


class SearchR1Env(ToolEnvironment):
    def __init__(self, task: dict | None = None, tools: list[str] | None = None, tool_map: dict[str, type[Tool]] | None = None, reward_fn: RewardFunction | None = None, max_steps=10):
        super().__init__(task=task, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn)
        self.input_args = {
            'task': task,
            'tools': tools,
            'tool_map': tool_map,
            'max_steps': max_steps,
            'reward_fn': reward_fn
        }
        self.format_reward = 0.0
        self.cache = {}

    def step(self, llm_response: list[dict] | str | dict):
        """
        Take a step in the environment based on the action.

        Args:
            actions: List containing a single action string from the agent

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if llm_response is None:
            llm_response = []
        assert isinstance(llm_response, dict)
        self.step_count += 1
        reward = 0
        action, step_format_reward = self.parse(llm_response['response'])
        # self.format_reward = min(self.format_reward, step_format_reward)
        done = (action is None or 'answer' in action)
        if done:
            llm_response = llm_response['response']
            task_info = self.task if self.task is not None else {}
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            msgs = self.task.get('messages', None)
            if self.step_count == 1:
                if not msgs or len(msgs) == 1: 
                    step_format_reward = - 0.5
            reward = reward_output.reward
            return {}, reward, done, {'format_reward': step_format_reward}
        elif 'search' in action:
            cache_result = self.get_cache(action['search'])
            if cache_result is not None:
                next_obs = cache_result
            else:
                retries = 5
                for _ in range(retries):
                    tool_id = str(uuid.uuid4())
                    tool_calls = [
                        {"function":
                            {
                                "name": "local_search",
                                "arguments":json.dumps({
                                    "query": action['search'],
                                    "top_k": 3
                                })
                            },
                        "id": tool_id
                        }
                    ]
                    tool_outputs = self._execute_tool_calls(tool_calls)
                    output = tool_outputs[tool_id]
                    if not output.startswith("Error"):
                        break
                next_obs = f"<information>{output}</information>"
                self.insert_cache(action['search'], next_obs)
            # Return results as lists with single items to maintain batch structure
            return next_obs, reward, done, {'format_reward': step_format_reward}
        else:
            next_obs = f"<information>Please provide the valid output.</information>"
            return next_obs, 0.0, done, {'format_reward': step_format_reward}

    def insert_cache(self, query, result):
        self.cache[query] = result
    
    def get_cache(self, query):
        if query in self.cache:
            return self.cache[query]
        return None

    def close(self):
        del self.cache
        super().close()
        return

    @staticmethod
    def parse(content: str):
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)

        tag_type = None
        if answer_match and (not search_match or answer_match.start() < search_match.start()):
            tag_type = 'answer'
        elif search_match:
            tag_type = 'search'

        extracted_content = None
        if tag_type == 'answer':
            extracted_content = {'answer': answer_match.group(1)}
        elif tag_type == 'search':
            extracted_content = {'search': search_match.group(1)}
        else:
            return None, -0.5  

        pattern = r'^\s*<thinking>(.*?)</thinking>\s*(<answer>.*?</answer>|<search>.*?</search>)\s*$'
        full_match = re.fullmatch(pattern, content, re.DOTALL)

        if not full_match:
            format_reward = -0.5
        else:
            tag_str = full_match.group(2)
            if ('<answer>' in tag_str and '<search>' in tag_str) or \
            tag_str.count('<answer>') > 1 or tag_str.count('<search>') > 1:
                format_reward = -0.5
            else:
                format_reward = 0.0

        return extracted_content, format_reward    

    @staticmethod
    def from_dict(env_args: dict) -> "SearchR1Env":
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        return SearchR1Env(task=env_args, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn)
