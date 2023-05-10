import torch
import transformers
from datasets import load_dataset

import os
import sys
from typing import List

import peft
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,)

from utils.prompter import Prompter
from transformers import LlamaForCausalLM, LlamaTokenizer

# You will have to create your own account and login
import wandb
wandb.login()


# In[ ]:


# model/data params
base_model: str = "/ibex/user/radhaks/LLMs/LLaMA_7B/Alpaca"  # the only required argument
data_path: str = "/ibex/user/radhaks/LLMs/LLaMA_7B/alpaca-lora-srijith/data/train_data_all_three.json" # alpaca_data_cleaned_archive.json"
wandb_run_name: str = "5e-5"

output_dir: str = os.path.join("/ibex/user/radhaks/LLMs/LLaMA_7B/alpaca-lora-srijith/weights", wandb_run_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# training hyperparams
batch_size: int = 128
micro_batch_size: int = 4 # batch size per GPU 
num_epochs: int = 5
learning_rate: float = 5e-5
cutoff_len: int = 2000 # will have to change this
val_set_size: int = 100 # amount of val datapoints to steal from your train datapoints
    
# lora hyperparams - did not change
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: List[str] = ["q_proj","v_proj",]
    
# llm hyperparams LOT of mistakes here as these are totally ignored in the funtions below - check out alpaca code
train_on_inputs: bool = True  # if False, masks out inputs in loss
add_eos_token: bool = True # was false COULD CAUSE POTENTIAL PROBLEMS
group_by_length: bool = False  # faster, but produces an odd training loss curve
# wandb params
wandb_project: str = "Alpaca_Lora_fine-tuning"
wandb_watch: str = "false"  # options: false | gradients | all
wandb_log_model: str = "true"  # options: false | true
resume_from_checkpoint: str = None  # either training checkpoint or final adapter
prompt_template_name: str = "alpaca"  # The prompt template to use, will default to alpaca.

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        # THINGS TO DO 1
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
    )
assert (
    base_model
), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
gradient_accumulation_steps = batch_size // micro_batch_size

# does not apply here - our code template is different
prompter = Prompter(prompt_template_name)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

# Check if parameter passed or if set within environ
use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)

# Only overwrite environ if wandb param passed
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
if len(wandb_watch) > 0:
    os.environ["WANDB_WATCH"] = wandb_watch
if len(wandb_log_model) > 0:
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model


# the model parameters are frozen anyway 
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

tokenizer = LlamaTokenizer.from_pretrained(base_model)

tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token -check alpaca code first
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):

# tokenize the prompt witout padding and add eos_token. Also add a dictionary field 
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token): # 
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    
    # full_prompt = prompter.generate_prompt(
    #     data_point["instruction"],
    #     data_point["input"],
    #     data_point["output"],
    # )
    
    full_prompt = data_point['prompt']
    
    tokenized_full_prompt = tokenize(full_prompt)
    # if False:#not train_on_inputs:  # all your promts have inputs
    #     user_prompt = prompter.generate_prompt(
    #         data_point["instruction"], data_point["input"]
    #     )
    #     tokenized_user_prompt = tokenize(
    #         user_prompt, add_eos_token=add_eos_token
    #     )
    #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #     if add_eos_token:
    #         user_prompt_len -= 1

    #     tokenized_full_prompt["labels"] = [
    #         -100
    #     ] * user_prompt_len + tokenized_full_prompt["labels"][
    #         user_prompt_len:
    #     ]  # could be sped up, probably
    return tokenized_full_prompt


# moving all the data stuff here
data_path = '/ibex/user/radhaks/LLMs/LLaMA_7B/alpaca-lora-srijith/data/train_data_all_three.json'
if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    data = load_dataset("json", data_files=data_path)
else:
    data = load_dataset(data_path)
    
if val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

# Had to make chages to the funtion to get rid of RuntimeError: expected scalar type Half but found Float 
def prepare_model_for_int8_training__(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float16)

    if loaded_in_8bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


model = prepare_model_for_int8_training__(model) 

# for name, param in model.named_parameters():
#     # freeze base model's layers
#     param.requires_grad = False
#     if getattr(model, "is_loaded_in_8bit", False):
#         if param.ndim == 1 and "layer_norm" in name:
#             param.data = param.data.to(torch.float16)



# model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.enable_input_require_grads()



config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=2,
        eval_steps=2 if val_set_size > 0 else None,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        save_steps=30,
        output_dir=output_dir,
        save_total_limit=10,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name= wandb_run_name if use_wandb else None,),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)
model.config.use_cache = False

old_state_dict = model.state_dict

model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

model.save_pretrained(output_dir)

print("\n If there's a warning about missing keys above, please disregard :)")

wandb.finish()