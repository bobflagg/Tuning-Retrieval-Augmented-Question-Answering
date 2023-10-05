from peft import LoraConfig, PeftModel
import textwrap
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_base_model(model_name, pad_token = "<PAD>", padding_side = "right", eos_token = "</s>"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = pad_token
    tokenizer.padding_side = padding_side
    tokenizer.eos_token = eos_token
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return tokenizer, model, generator

def load_lora(model_name, directory, pad_token = "<PAD>", padding_side = "right", eos_token = "</s>", model=None):    
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    model = PeftModel.from_pretrained(model, directory)
    model = model.merge_and_unload()
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = pad_token
    tokenizer.padding_side = padding_side
    tokenizer.eos_token = eos_token
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return tokenizer, model, generator

import random
import textwrap
import time

def extract_prompt(sample):
    text = sample['text']
    index = text.find("[/INST]")
    prompt = text[:index + 8] + "\n\n%%ANSWER:"
    return prompt
    
def generate(sample, generator):
    prompt = extract_prompt(sample)
    completion = generator(prompt, max_new_tokens=200)[0]['generated_text']
    completion = completion.split("%%ANSWER: ")[-1].strip()
    return completion
    
def display_completion(sample, generator):
    completion = generate(sample, generator=generator)
    display_sample(sample, response=completion, max_width=120)
    return completion
    
def build_prompt(x, prefix):
    return f"<s>[INST] {prefix}\n\n%%EXCERPT: {x.excerpt}\n\n%%QUESTION: {x.question}[/INST]\n\n%%ANSWER:"   

def display_sample(x, prefix=None, generator=None, max_width=120, max_new_tokens=256):
    def print_boxed(text):
        lines = textwrap.wrap(text, max_width)  # Wrap text to desired width
        border = '+' + '-' * (max_width + 2) + '+'
        print(border)
        for line in lines:
            print('| ' + line.ljust(max_width) + ' |')
        print(border)
    print("INSTRUCTIONS:")
    print_boxed(prefix)
    print("\nEXCERPT:")
    print_boxed(x.excerpt)
    print("\nQUESTION:")
    print_boxed(x.question)
    if generator is not None:
        prompt = build_prompt(x, prefix)
        start = time.time()        
        completion = generator(prompt, max_new_tokens=max_new_tokens)[0]['generated_text']
        end = time.time()
        delta = end - start
        answer = completion.split("%%ANSWER:")[-1].strip()
        print(f"\nGENERATED-ANSWER ({delta:0.3f}):")
        print_boxed(answer)
    print("\nGOLD-ANSWER:")
    print_boxed(x.answer)
