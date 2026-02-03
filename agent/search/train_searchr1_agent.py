import hydra

from rllm.agents.search_r1_agent import SearchR1Agent
from rllm.data import DatasetRegistry
from rllm.environments.search_r1.search_r1 import SearchR1Env
from rllm.rewards.reward_fn import search_r1_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from .local_retrieval_tool import LocalRetrievalTool


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("search_r1", config.data.train_data)
    val_dataset = DatasetRegistry.load_dataset("search_r1", config.data.test_data)

    tool_map = {"local_search": LocalRetrievalTool}

    env_args = dict(config.rllm.env.get('env_args', {}))
 
    env_args["max_steps"] = env_args.get('max_steps', 4)

    env_args["tool_map"] = tool_map
    env_args["reward_fn"] = search_r1_reward_fn


    agent_args = {}

    trainer = AgentTrainer(
        agent_class=SearchR1Agent,
        env_class=SearchR1Env,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
