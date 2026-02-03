
<div align="center">

# BranPO: Training Multi-Turn Search Agent via Contrastive Branch Sampling

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

Next, download the [ASearcher](https://github.com/inclusionAI/ASearcher/tree/main) local retrieval server and retriever:

```bash
hf download inclusionAI/ASearcher-Local-Knowledge --repo-type dataset
hf download intfloat/e5-base-v2
```

Finally, build the index:

```bash
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

We provide scripts for both **GRPO** and **BranPO** in `./train_grpo.sh` and `./train_branpo.sh`, respectively.


Make sure you have updated the model paths and retrieval knowledge base paths to your local directories before starting.


## 📊 Evaluation

To evaluate your model, run `run_eval.sh` to test against the local retrieval server. Following that, execute `run_llm_as_a_judge.sh` to perform the LLM-as-a-Judge evaluation.

## 🤝 Acknowledgements

This codebase is built upon [rLLM](https://github.com/rllm-org/rllm) and [veRL](https://github.com/verl-project/verl). The search workflow and training data are based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [ASearcher](https://github.com/inclusionAI/ASearcher). We are sincerely grateful to these projects for their foundational contributions to the field!
