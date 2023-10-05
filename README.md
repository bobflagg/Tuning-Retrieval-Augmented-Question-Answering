# Tuning Retrieval Augmented Question Answering

In this repo, we will design, run and analyze experiments to determine critical factors that influence the quality of answers in retrieval augmented question answering with large language models.  We will consider the effect of the LLM used, the prompting strategy employed, and the use of domain-specific fine-tuning on the model's performance. The goal is to optimize these factors to achieve the highest proportion of correctly answered questions.

## Fine-tuning LLaMA-2-7B

The [training script](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/scripts/train.py) can be used for supervised fine-tuning of open source models from Hugging Face Hub with QLoRA for question answering over sources.  For example, to fine-tune LLaMA-2-7B on the small demo dataset included in this repo, run

```bash
python train.py --fp16
```

If the base model weights are not already available on your system, you'll first need to login to the hub:
```bash
huggingface-cli login --token hf_...
```

This will train the model for one epoch on records having folds 0 - 9 so evaluation can be done on records with fold 10 and the fine-tuned adpater weights will be saved in ./model/Llama-2-7b-qa-10. Training takes about 15 minutes on an NVIDIA A10G.   You can review sample inferences with the model in the [Review-Generated-Answers notebook](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/notebook/Review-Generated-Answers.ipynb).

#
# huggingface-cli login --token hf_...
#
# To review generated answers for the base and fine-tuned models, see ../notebook/Review-Generated-Answers.ipynb.
#
# The key requirements to run this script are
# 
#   accelerate==0.21.0
#   bitsandbytes==0.40.2
#   huggingface_hub==0.17.1
#   peft==0.4.0
#   transformers==4.31.0
#   trl==0.4.7
# 
# The code has been testing on an AWS EC2 g5.4xl instance with an NVIDIA A10G having 24GB GPU memory
# but should run fine on a GPU having 16GB of memory.
# 
# This code is based on 
#      https://github.com/dstackai/dstack-examples/blob/main/llama-2/train.py
# with minor modifications to simplify data loading.
#

 1. Login to the [Hugging Face Hub](https://huggingface.co/docs/hub/index):

```
huggingface-cli login --token hf_...
```

 2. Update the ROOT environment variable in [train.sh](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/scripts/train.sh)
 3. Run the trainging shell script; for example, to train on all but the 10th fold, run

```
./train.sh 10
```

Training takes about 15 minutes on an NVIDIA A10G. The fine-tuned adpater weights will be saved in $ROOT/Llama-2-7b-qa-10.  You can review sample inferences with the model in the [Review-Generated-Answers notebook](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/notebook/Review-Generated-Answers.ipynb).

## Sample Dataset

A small [dataset](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/data/train-test-df.csv) with excerpts, questions and synthetically generated answers is available for training and evaluation of QA with LLMs.  The **fold** field of this dataset has values 0 - 10.  We use folds 1 - 10, which have 100 records each, for replicating runs in hypothesis testing.  



