import json
import os
import datasets
import argparse
from rllm.data.dataset import DatasetRegistry

def prepare_train_data():
    data_sources = ['nq', 'hotpotqa']
    all_data = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        train_dataset = dataset['train']

        for example in train_dataset:
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'

            ground_truth = example['golden_answers']

            item = {
                'question': question,
                'ground_truth': ground_truth,
                'data_source': data_source,
            }
            all_data.append(item)
    train_dataset = DatasetRegistry.register_dataset("search_r1", all_data, "train1")
    return train_dataset.get_data()

def prepare_test_data():
    data_sources = "nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle".split(',')
    all_data = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        for example in test_dataset:
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'

            ground_truth = example['golden_answers']

            item = {
                'question': question,
                'ground_truth': ground_truth,
                'data_source': data_source,
            }
            all_data.append(item)

    test_dataset = DatasetRegistry.register_dataset("search_r1", all_data, "test1")
    return test_dataset.get_data()

if __name__ == "__main__":
    train_dataset = prepare_train_data()
    print(f"Train dataset: {train_dataset[0]}")

    test_dataset = prepare_test_data()
    print(f"Test dataset: {test_dataset[0]}")