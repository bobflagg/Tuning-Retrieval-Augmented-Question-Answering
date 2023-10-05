# Tuning Retrieval Augmented Question Answering

In this repo, we will design, run and analyze experiments to determine critical factors that influence the quality of answers in retrieval augmented question answering with large language models.  We will consider the effect of the LLM used, the prompting strategy employed, and the use of domain-specific fine-tuning on the model's performance. The goal is to optimize these factors to achieve the highest proportion of correctly answered questions.

## Fine-tuning LLaMA-2-7B

The [training script](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/scripts/train.py) can be used for supervised fine-tuning of open source models from Hugging Face Hub with QLoRA for question answering over sources.  For example, to fine-tune LLaMA-2-7B on the small demo dataset included in this repo, after logging into the hub with

```
huggingface-cli login --token hf_...
```
run
```bash
python train.py --fp16
```

This will train the model for one epoch on records having folds 0 - 9 so evaluation can be done on records with fold 10.  The adpater weights will be saved in ./model/Llama-2-7b-qa-10. Training takes about 15 minutes on an NVIDIA A10G.   Sample inferences with the model can be run in the [Review-Generated-Answers notebook](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/notebook/Review-Generated-Answers.ipynb).

The key requirements to run this script are

    - accelerate==0.21.0
    - bitsandbytes==0.40.2
    - huggingface_hub==0.17.1
    - peft==0.4.0
    - transformers==4.31.0
    - trl==0.4.7

This has been tested on an AWS EC2 g5.4xl instance with an NVIDIA A10G having 24GB GPU memory but the script should run fine on a GPU having 16GB of memory.
 

This code is based on the [dstack sample](https://github.com/dstackai/dstack-examples/blob/main/llama-2/train.py) with minor modifications to simplify data loading.



