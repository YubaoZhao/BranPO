import copy
import logging
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
f"""Answer the given question. \
You must conduct reasoning inside <thinking> and </thinking> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """
)

class SearchR1Agent(BaseAgent):
    def __init__(
        self,
        system_prompt=SYSTEM_PROMPT,
    ):
        # Initialize state according to BaseAgent
        self.system_prompt = system_prompt
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.tokens: list[int] = []
 
        self.reset()  # Call reset to set initial state

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Helper to format observation into messages."""
        messages = []
        if isinstance(obs, dict):
            if "messages" in obs and len(obs['messages']) > 0:
                messages = obs['messages']
            elif "question" in obs:
                messages.append({"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + obs["question"]})
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        assistant_content = response
        tool_calls_dict = {'response': assistant_content}
        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": assistant_content}

        self.messages.append(assistant_message)

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), action=tool_calls_dict, model_response=response, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)

    def parser(self, response: str) -> dict:
        search_match = re.search(r'<search>\s*(.*?)\s*</search>', response, re.DOTALL)
        if search_match:
            return {'search': search_match.group(1).strip()}
        return {}

    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None
        self.tokens: list[int] = []

    def update_tokens(self, tokens):
        self.tokens.extend(tokens)

    @property
    def chat_completions_tokens(self):
        return self.tokens

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory

