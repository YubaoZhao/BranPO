from datasets import load_dataset
import json
from rllm.data.dataset import DatasetRegistry
import os

TRAIN_DATA_PATH="/path/to/ASearcher-Base-35k.jsonl"
TEST_DATA_PATH="/path/to/ASearcher-test-dir"

def prepare_asearcher_train_data():
    get_answer = lambda ex: ex['aug_answer'] if ex['answer'][0] in ex['aug_answer'] else ex['answer'] + ex['aug_answer']
    def process(example):
        return {'question': example["question"], "ground_truth": get_answer(example), "data_source": "asearcher-base"}
    data = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f.readlines():
            data.append(process(json.loads(line)))
    filtered_data = []
    for item in data:
        if 'question' not in item['ground_truth'][0]:
            filtered_data.append(item)
    print("Number of dataset: ", len(filtered_data))
    train_dataset = DatasetRegistry.register_dataset("search_r1", filtered_data, "train-asearcher")
    
    return train_dataset.get_data()

def prepare_asearcher_test_data():
    dir = TEST_DATA_PATH
    data_names = ['Bamboogle','PopQA_rand1000','2WikiMultihopQA_rand1000','NQ_rand1000','TriviaQA_rand1000','HotpotQA_rand1000','Musique_rand1000']
    test_data = []
    for name in data_names:
        dir_path = os.path.join(dir, name)
        json_path = os.path.join(dir_path, 'test.json')
        jsonl_path = os.path.join(dir_path, 'test.jsonl')
        
        # determine .json or .jsonl
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            print(f"Warning: No test.json or test.jsonl found in {dir_path}")
            continue

        for item in data:
            question = item.get('question')
            if question is None:
                continue 

            answer = item.get('answer') or item.get('gt')
            if answer is None:
                ground_truth = []
            elif isinstance(answer, str):
                ground_truth = [answer]
            elif isinstance(answer, list):
                ground_truth = [str(a) for a in answer if isinstance(a, (str, int, float))]
            else:
                ground_truth = [str(answer)]
            
            test_data.append({
                'question': question,
                'ground_truth': ground_truth,
                'data_source': name
            })
    test_dataset = DatasetRegistry.register_dataset("search_r1", test_data, "test-asearcher")
    return test_dataset.get_data()


def prepare_gaia():
    dir = TEST_DATA_PATH
    test_data = []
    name = 'GAIA'
    dir_path = os.path.join(dir, name)
    json_path = os.path.join(dir_path, 'test.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        question = item.get('question')
        if question is None:
            continue 

        answer = item.get('answer')
        if answer is None:
            ground_truth = []
        elif isinstance(answer, str):
            ground_truth = [answer]
        elif isinstance(answer, list):
            ground_truth = [str(a) for a in answer if isinstance(a, (str, int, float))]
        else:
            ground_truth = [str(answer)]
        
        test_data.append({
            'question': question,
            'ground_truth': ground_truth,
            'data_source': item.get('source')
        })
    test_dataset = DatasetRegistry.register_dataset("search_r1", test_data, "test-gaia")
    return test_dataset.get_data()


if __name__ == "__main__":
    train_dataset = prepare_asearcher_train_data()
    print(f"Train dataset: {train_dataset[0]}")

    test_dataset = prepare_asearcher_test_data()
    print(f"Test dataset: {test_dataset[0]}")

    test_gaia = prepare_gaia()
    print(f"GAIA Test dataset: {test_gaia[0]}")

