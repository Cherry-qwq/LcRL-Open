# LcRL

## Installation

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```


## Quick start

Train a reasoning + search LLM on MKQA dataset with multilingual-e5 as the retriever and wikipedia as the corpus.

(1) Download the corpus in https://huggingface.co/datasets/wikimedia/wikipedia, and index the corpus.
```bash
bash search_r1/search/m_build_index_parallel.sh
```

(2) Download your dataset, and then process the multilingual train and test dataset.
```bash
python scripts/data_process/mkqa_search_parallel.py
```
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

(3) Launch a local retrieval server.
```bash
conda activate retriever
bash m_retrieval_launch_parallel.sh
#The LLM can call the search engine by calling the search API (e.g., "http://127.0.0.1:8000/retrieve").
```

(4) Run RL training (GRPO) .
```bash
conda activate searchr1
bash m_train_grpo.sh
```

(5) View results .
```bash
mlflow ui
```