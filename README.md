
<div align="center">

# BranPO: Scalable Contrastive Branch Sampling for Long-Horizon Agentic Reinforcement Learning

</div>
<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.03719)
</div>

## 🚀 Get Started

### Search Engine Construction

First, set up the local search environment by following [Search-R1](https://github.com/PeterGriffinJin/Search-R1):

```bash
conda create -n retriever python=3.10
conda activate retriever

# We recommend installing torch with conda for faiss-gpu compatibility
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

# Install the GPU version of faiss to ensure efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# API dependencies
pip install uvicorn fastapi

```

Next, download the index and corpus:
```bash
save_path=/the/path/to/save
python agent/search/retrieval/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

For reproducing the results of the Asearcher dataset, please download the [ASearcher](https://github.com/inclusionAI/ASearcher/tree/main) local retrieval server and retriever, then build the index:

```bash
hf download inclusionAI/ASearcher-Local-Knowledge --repo-type dataset
hf download intfloat/e5-base-v2

bash agent/search/retrieval/build_index.sh
```

### Training Environment

Set up the environment for RL training:

```bash
conda create -n rllm python=3.10
cd ./BranPO/
pip install -e .

```

### Data Preparation
**(1) Search-R1**
Run `agent/search/prepare_searchr1_data.py`:
```python
python agent/search/prepare_searchr1_data.py
```

**(2) ASearcher**
Download the ASearcher training and test datasets:

```bash
hf download inclusionAI/ASearcher-train-data --repo-type dataset
hf download inclusionAI/ASearcher-test-data --repo-type dataset
```

After downloading, update the dataset file paths in `agent/search/prepare_asearcher_data.py` to match your local directories, then run the script to preprocess the data.


## 🏋️ Training

### Cold Start

The 10k SFT cold start dataset is available on [Hugging Face](https://huggingface.co/datasets/ThornZ/Search-R1-SFT):

```bash
hf download ThornZ/Search-R1-SFT --repo-type dataset
```

We recommend using [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) for SFT training. You can find the provided training scripts in the `sft/` directory.

### RL Training

We provide scripts for both **GRPO** and **BranPO** in `scripts/`.


Make sure you have updated the model paths and retrieval knowledge base paths to your local directories before starting.


## 📊 Evaluation

To evaluate your model, run `scripts/eval/run_eval.sh` to test against the local retrieval server. Following that, execute `scripts/eval/run_llm_as_a_judge.sh` to perform the LLM-as-a-Judge evaluation.

## 🏆 Preliminary Results
Using Qwen2.5-7B-Instruct as the backbone model, and conduct BranPO training on Search-R1 dataset, the F1 score on 4 multi-hop datasets are shown as below:

| Method | 2WikiMQA | HotpotQA | MuSiQue | Bamboogle | Average |
|---|---:|---:|---:|---:|---:|
| Search-R1 | 37.6 | 50.2 | 26.2 | 50.1 | 41.0 |
| SE-Search | 42.2 | 55.9 | 29.0 | 60.1 | 46.8 |
| ReasonRAG | 50.4 | 48.9 | 20.6 | 45.5 | 41.3 |
| StepSearch | 43.1 | 50.2 | 31.2 | 53.4 | 44.5 |
| GiGPO | 48.0 | 53.4 | 27.1 | 50.1 | 44.7 |
| CriticSearch | 50.1 | 56.0 | 28.1 | 59.2 | 48.4 |
| TIPS | 50.6 | 54.7 | 26.6 | 52.2 | 46.0 |
| Tree-GRPO | 48.6 | 56.2 | 30.3 | 57.1 | 48.1 |
| ARPO | 51.3 | 59.1 | 27.3 | 50.9 | 47.2 |
| AEPO | 53.5 | 58.2 | 28.4 | 54.6 | 48.7 |
| **BranPO** | **56.7** | **61.8** | **32.4** | **62.0** | **53.2** |

## 🤝 Acknowledgements

This codebase is built upon [rLLM](https://github.com/rllm-org/rllm) and [veRL](https://github.com/verl-project/verl). The search workflow and training data are based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [ASearcher](https://github.com/inclusionAI/ASearcher). We are sincerely grateful to these projects for their foundational contributions to the field!

## Citation
```
@article{zhao2026branpo,
  title={BranPO: Scalable Contrastive Branch Sampling for Long-Horizon Agentic Reinforcement Learning},
  author={Zhao, Yubao and Huang, Weiquan and Wang, Sudong and Zhao, Ruochen and Chen, Chen and Shu, Yao and Qin, Chengwei},
  journal={arXiv preprint arXiv:2602.03719},
  year={2026}
}
```