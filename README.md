# Tuning Retrieval Augmented Question Answering

In this repo, we will design, run and analyze experiments to determine critical factors that influence the quality of answers in retrieval augmented question answering with large language models.  We will consider the effect of the LLM used, the prompting strategy employed, and the use of domain-specific fine-tuning on the model's performance. The goal is to optimize these factors to achieve the highest proportion of correctly answered questions.

## Fine-tuning LLaMA-2-7B

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



