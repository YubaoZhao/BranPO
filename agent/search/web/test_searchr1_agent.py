import argparse
import asyncio
import os

from dotenv import load_dotenv
from web_search_tool import WebRetrievalTool
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.agents.search_r1_agent import SearchR1Agent
from rllm.environments.search_r1.search_r1 import SearchR1Env
from rllm.rewards.reward_fn import search_r1_reward_fn
from rllm.utils import save_trajectories
import torch
import copy
from collections import defaultdict

def compute_pass_at_k(results):
    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    problem_score_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task
        problem_idx = trajectory.task['index']
        metadata = search_r1_reward_fn(task, trajectory.steps[-1].chat_completions[-1]['content']).metadata
        try:
            is_correct = 1 if metadata['exact_match'] else 0
        except:
            is_correct = 0

        problem_correct_map[problem_idx] += is_correct
        problem_total_map[problem_idx] += 1
        try:
            problem_score_map[problem_idx] += metadata['f1_score']
        except:
            problem_score_map[problem_idx] += 0

    # Calculate pass@1 and pass@k
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems
    avg_at_k = sum(problem_score_map.values()) / len(results)
    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)
    print("Average F1 score:", avg_at_k)

def load_test_data(data_name='test', k=1):
    """
    Load search data, preparing it if not already available.
    Returns the test dataset data for evaluation.
    """
    test_dataset = DatasetRegistry.load_dataset("search_r1", data_name)
    for i in range(len(test_dataset)):
        test_dataset[i]['index'] = i
    test_dataset = test_dataset.repeat(k)
    data = test_dataset.get_data()
    return data

def main():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default='test')
    parser.add_argument("--pass_at_k", type=int, default=1)
    parser.add_argument("--agent_max_steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--dp_size", type=int, default=4)
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    load_dotenv()

    n_parallel_agents = 2048
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": args.temperature, "top_p": args.top_p, "model": model_name}

    tool_map = {"web_search": WebRetrievalTool}

    env_args = {
        "tool_map": tool_map,
        "reward_fn": search_r1_reward_fn,
    }


    engine = AgentExecutionEngine(
        agent_class=SearchR1Agent,
        agent_args={},
        env_class=SearchR1Env,
        env_args=env_args,
        rollout_engine=None,
        engine_name="vllm",
        max_steps=args.agent_max_steps,
        trajectory_timeout=1800,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "model": model_name,
            "tensor_parallel_size": args.tp_size,
            "data_parallel_size": args.dp_size,
            "gpu_memory_utilization": 0.8
        },
        max_prompt_length=1024,
        max_response_length=int(args.max_model_len)-1024,
        config=None,
        n_parallel_agents=n_parallel_agents,
    )

    tasks = load_test_data(args.test_data, args.pass_at_k)
    results = asyncio.run(engine.execute_tasks(tasks))

    save_trajectories(results, filename=f"{args.run_name}-{args.test_data}-ms{args.agent_max_steps}-t{args.temperature}-top_p{args.top_p}-pass_at_{args.pass_at_k}.pt")
    source_results = {}

    for ret in results:
        source = ret.task['data_source']
        if source not in source_results.keys():
            source_results[source] = []
        source_results[source].append(ret)
    
    for key in source_results.keys():
        print("="*20)
        print("TEST SET: ", key)
        rets = source_results[key]
        compute_pass_at_k(rets)
  

if __name__ == "__main__":
    main()
