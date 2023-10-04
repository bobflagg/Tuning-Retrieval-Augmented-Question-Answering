# Tuning Retrieval Augmented Question Answering

In this repo, we will design, run and analyze experiments to determine critical factors that influence the quality of answers in retrieval augmented question answering with large language models.  We will consider the effect of the LLM used, the prompting strategy employed, and the use of domain-specific fine-tuning on the model's performance. The goal is to optimize these factors to achieve the highest proportion of correctly answered questions.

## Sample Dataset

A small [dataset](https://github.com/bobflagg/Tuning-Retrieval-Augmented-Question-Answering/blob/main/data/train-test-df.csv) with excerpts, questions and synthetically generated answers is available for training and evaluation of QA with LLMs.  The **fold** field of this dataset has values 0 - 10.  We use folds 1 - 10, which have 100 records each, for replicating runs and hypothesis testing.  



