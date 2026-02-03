import asyncio
import json
import os
import re
import argparse
from typing import List, Dict, Any, Optional, Set, Union

import torch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


class DefaultJudge:
    JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to any of the labeled answers.

Question: {question}

Labeled Answers: {gt_answers_str}

Predicted Answer: {pred_answer}

Did the model give an answer **equivalent** to any of the labeled answers? Respond with "Correct" if equivalent to at least one, otherwise "Incorrect".

Output in JSON format:
```json
{{
    "rationale": "your rationale",
    "judgement": "Correct" or "Incorrect"
}}
```"""

    @staticmethod
    def parse_judgement(raw_response: str) -> str:
        """
        Parses the JSON response from the LLM and extracts the judgement.
        """
        try:
            # Attempt to extract JSON from markdown code blocks
            json_str = raw_response.split("```json")[-1].split("```")[0].strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "judgement" in parsed:
                judgement = parsed["judgement"].strip().lower()
                if judgement == "correct":
                    return "Correct"
                elif judgement == "incorrect":
                    return "Incorrect"
        except Exception:
            pass

        # Fallback: Simple string matching if JSON parsing fails
        lower_resp = raw_response.lower()
        if '"judgement": "correct"' in lower_resp:
            return "Correct"
        if '"judgement": "incorrect"' in lower_resp:
            return "Incorrect"

        return "Invalid"


async def llm_as_judge_batch_openai(
    samples: List[Dict[str, Any]],
    output_file: str,
    model: str,
    api_key: str,
    base_url: str,
    max_concurrent: int = 10,
    max_retries: int = 5,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs batch LLM evaluation with concurrency control and checkpointing.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)
    judge = DefaultJudge()

    # Load checkpointed results to skip already processed IDs
    completed_ids: Set[str] = set()
    results_map: Dict[str, Dict] = {}

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        completed_ids.add(str(item["id"]))
                        results_map[str(item["id"])] = item
                    except json.JSONDecodeError:
                        continue
        print(f"[INFO] Loaded {len(completed_ids)} existing results from {output_file}")

    pending_samples = [s for s in samples if str(s["id"]) not in completed_ids]
    new_results = []

    def format_gt(gt_list: List[str]) -> str:
        if len(gt_list) == 1:
            return f'"{gt_list[0]}"'
        return "[" + ", ".join(f'"{a}"' for a in gt_list) + "]"

    async def evaluate_single(sample: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            gt_str = format_gt(sample["ground_truth"])
            prompt = judge.JUDGE_PROMPT.format(
                question=sample["question"],
                gt_answers_str=gt_str,
                pred_answer=str(sample.get("pred_answer", ""))[:1000], # Cap length
            )

            retries = 0
            raw_response = ""
            judgement = "Invalid"

            while retries <= max_retries:
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    raw_response = completion.choices[0].message.content or ""
                    judgement = judge.parse_judgement(raw_response)
                    if judgement != "Invalid":
                        break
                except Exception as e:
                    if retries == max_retries:
                        print(f"[ERROR] Final failure for id={sample['id']}: {e}")
                    await asyncio.sleep(1 + retries * 0.5)
                retries += 1

            result = {
                "id": sample["id"],
                "judgement": judgement,
                "raw_response": raw_response,
                "retries": retries,
            }

            # Immediate file append for safety (checkpointing)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            return result

    # Execute pending tasks
    if pending_samples:
        tasks = [evaluate_single(sample) for sample in pending_samples]
        batch_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        new_results.extend(batch_results)
    else:
        print("[INFO] No pending samples found.")

    # Merge all results
    final_results_dict = {str(r["id"]): r for r in new_results}
    final_results_dict.update(results_map)

    # Group results by data_source for metrics
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        sid = str(sample["id"])
        source = sample.get("data_source", "unknown")
        
        if source not in grouped_results:
            grouped_results[source] = []
        
        # Append merged result data
        if sid in final_results_dict:
            res_item = {**sample, **final_results_dict[sid]}
            grouped_results[source].append(res_item)
            
    return grouped_results


def extract_answer(traj: Any) -> str:
    """
    Extracts the content between <answer> tags from the last assistant message.
    """
    try:
        last_assistant_msg = traj.steps[-1].chat_completions[-1]["content"]
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", last_assistant_msg, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else "No answer found"
    except (AttributeError, IndexError):
        return "Parsing Error"


async def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation Script")
    parser.add_argument("--names", nargs="+", required=True, help="List of trajectory names (filenames) to evaluate")
    parser.add_argument("--input_dir", type=str, default="./trajectories", help="Directory containing .pt files")
    parser.add_argument("--output_dir", type=str, default="./llm_judge", help="Directory to save evaluation results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key for the judge")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL"), help="API Base URL")
    parser.add_argument("--max_concurrent", type=int, default=1024, help="Max concurrent requests")
    
    args = parser.parse_args()

    for traj_name in args.names:
        input_path = os.path.join(args.input_dir, f"{traj_name}.pt")
        output_path = os.path.join(args.output_dir, f"{traj_name}.jsonl")
        
        if not os.path.exists(input_path):
            print(f"[SKIP] Input file not found: {input_path}")
            continue

        print(f"\n[START] Processing trajectory: {traj_name}")
        trajs = torch.load(input_path, weights_only=False)
        
        sample_data = []
        for idx, traj in enumerate(trajs):
            sample_data.append({
                "id": idx,
                "data_source": traj.task.get('data_source', 'default'),
                "question": traj.task.get('question', ''),
                "ground_truth": traj.task.get('ground_truth', []),
                "pred_answer": extract_answer(traj)
            })

        source_results = await llm_as_judge_batch_openai(
            samples=sample_data,
            output_file=output_path,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_concurrent=args.max_concurrent
        )

        # Print Statistics
        print(f"\n--- Results for {traj_name} ---")
        for source, results in source_results.items():
            correct = sum(1 for r in results if r["judgement"] == "Correct")
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            print(f"Source: {source} | Accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    asyncio.run(main())