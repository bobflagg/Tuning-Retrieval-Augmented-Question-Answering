# This script can be used for supervised fine-tuning of open source models from Hugging Face Hub 
# for question answering over sources.  For example, to fine-tune LLaMA-2-7B on the small demo
# dataset included in this repo and save the LoRA adaptor weights in the subdirectory ./models, 
# run
#
# python train.py --fp16
#
# This will train the model for one epoch on records having folds 0 - 9 so evaluation can be done on records with fold 10.
# If the base model weights are not already available on your system, you'll first need to login to the hub:
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
# This code is based on 
#      https://github.com/dstackai/dstack-examples/blob/main/llama-2/train.py
# with minor modifications to simplify data loading.
#

from dataclasses import dataclass, field
from datasets import Dataset
import os
import pandas as pd
from peft import LoraConfig, PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer
from typing import Optional, Union

@dataclass
class ScriptArguments:
    data_path: Optional[str] = field(
        default="../data/train-test-df.csv", metadata={"help": "Path to a CSV file with train/test data."}
    )
    prefix_path: Optional[str] = field(
        default="../data/level-1-prompt-prefix.txt", metadata={"help": "Path to a file with the prefix to append to train/test prompts."}
    )
    output_directory: Optional[str] = field(
        default="./models", metadata={"help": "The directory where fine-tuned models will be saved."}
    )
    fold: Optional[int] = field(
        default=10, metadata={"help": "The dataset fold to omit in training."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for training."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for evaluation."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of update steps to accumulate the gradients for."},
    )
    learning_rate: Optional[float] = field(
        default=4e-5, metadata={"help": "Initial learning rate (AdamW optimizer)."}
    )
    max_grad_norm: Optional[float] = field(
        default=0.3, metadata={"help": "Maximum gradient normal (gradient clipping)."}
    )
    weight_decay: Optional[int] = field(
        default=0.001,
        metadata={
            "help": "Weight decay to apply to all layers except bias/LayerNorm weights."
        },
    )
    lora_alpha: Optional[int] = field(
        default=64, metadata={"help": "Alpha parameter for LoRA scaling."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability for LoRA layers."}
    )
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "LoRA attention dimension."}
    )
    max_seq_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum sequence length to use."}
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    new_model_name: Optional[str] = field(
        default="Llama-2-7b-qa",
        metadata={
            "help": "The name under which to push the fine-tuned model to the Hugging Face Hub."
        },
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=-1, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=50, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=25, metadata={"help": "Log every X updates steps."}
    )
    output_dir: str = field(
        default="/opt/dlami/nvme/enrichment/results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the arguments --bf16 --bnb_4bit_compute_dtype bfloat16"
            )
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False
    # https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"

    return model, tokenizer

def create_and_prepare_trainer(model, tokenizer, dataset, args):
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="tensorboard",
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )
    return trainer

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    with open(args.prefix_path) as ifp: prefix = ifp.read()
    
    def load(fold, path, prefix, dtype):
        df = pd.read_csv(path, na_filter=False)
        def build_input(x):
           return f"<s>[INST] {prefix}\n\n%%EXCERPT: {x.excerpt}\n\n%%QUESTION: {x.question}[/INST]\n\n%%ANSWER: {x.answer}</s>"    
        if dtype == "train": fold_df = df[df.fold != fold]
        else:  fold_df = df[df.fold == fold]
        return {"text":[build_input(x) for index, x in fold_df.iterrows()]}
        
    dataset = Dataset.from_dict(load(fold=args.fold, path=args.data_path, prefix=prefix, dtype='train'))

    model, tokenizer = create_and_prepare_model(args)

    trainer = create_and_prepare_trainer(model, tokenizer, dataset, args)

    trainer.train()
    trainer.model.save_pretrained(f"{args.output_directory}/{args.new_model_name}-{args.fold:02d}")
