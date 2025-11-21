import os
import yaml
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import UnslothTrainer, UnslothTrainingArguments
from huggingface_hub import login
import argparse

import wandb
wandb.login()

from dotenv import load_dotenv
load_dotenv()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def continual_pretrain(config):
    """Phase 1: Continual pre-training on Army FM text"""
    print("="*80)
    print("PHASE 1: Continual Pre-training")
    print("="*80)
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['base_model'],
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
        token=os.environ.get('HF_TOKEN'),
    )
    
    # Apply LoRA for efficient CPT
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing=config['lora']['use_gradient_checkpointing'],
        random_state=config['training']['seed'],
        use_rslora=config['lora']['use_rslora'],
    )
    
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"Total parameters: {model.num_parameters():,}")
    
    # Load dataset
    dataset = load_dataset(
        config['data']['dataset_name'], 
        split=config['data']['split']
    )
    
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        return {"text": [example + EOS_TOKEN for example in examples["text"]]}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Training
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config['training']['max_seq_length'],
        dataset_num_proc=config['training']['dataset_num_proc'],
        args=UnslothTrainingArguments(
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            warmup_ratio=config['training']['warmup_ratio'],
            num_train_epochs=config['training']['num_train_epochs'],
            learning_rate=config['training']['learning_rate'],
            embedding_learning_rate=config['training']['embedding_learning_rate'],
            logging_steps=config['training']['logging_steps'],
            optim=config['training']['optim'],
            weight_decay=config['training']['weight_decay'],
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            seed=config['training']['seed'],
            output_dir=config['training']['output_dir'],
            report_to=config['training']['report_to'],
        ),
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Merge LoRA weights and save
    print("\nMerging LoRA weights...")
    model = FastLanguageModel.for_inference(model)
    merged_model = model.merge_and_unload()
    
    # Save to HuggingFace
    base_cpt_name = config['output']['base_cpt_model']
    print(f"\nSaving base+CPT model to: {base_cpt_name}")
    merged_model.push_to_hub(base_cpt_name, token=os.environ.get('HF_TOKEN'))
    tokenizer.push_to_hub(base_cpt_name, token=os.environ.get('HF_TOKEN'))
    
    return base_cpt_name

def add_instruction_residuals(config, base_cpt_model_name):
    """Phase 2: Add instruction tuning residuals using task arithmetic"""
    print("\n" + "="*80)
    print("PHASE 2: Adding Instruction Residuals")
    print("="*80)
    
    base_model_name = config['model']['base_model']
    instruct_model_name = config['model']['instruct_model']
    
    print(f"\nLoading models for residual computation...")
    print(f"  Base: {base_model_name}")
    print(f"  Instruct: {instruct_model_name}")
    print(f"  CPT: {base_cpt_model_name}")
    
    # Load original base model
    print("\n[1/4] Loading original base model...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
        token=os.environ.get('HF_TOKEN'),
    )
    base_state_dict = base_model.state_dict()
    del base_model
    torch.cuda.empty_cache()
    
    # Load original instruct model
    print("[2/4] Loading original instruct model...")
    instruct_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=instruct_model_name,
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
        token=os.environ.get('HF_TOKEN'),
    )
    instruct_state_dict = instruct_model.state_dict()
    del instruct_model
    torch.cuda.empty_cache()
    
    # Compute instruction residuals: Θ_residual = θ_instruct - θ_base
    print("[3/4] Computing instruction residuals...")
    instruction_residuals = {}
    for key in base_state_dict.keys():
        if key in instruct_state_dict:
            instruction_residuals[key] = instruct_state_dict[key] - base_state_dict[key]
            print(f"  Computed residual for: {key}")
    
    del base_state_dict, instruct_state_dict
    torch.cuda.empty_cache()
    
    # Load CPT model
    print("[4/4] Loading CPT model and adding residuals...")
    cpt_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_cpt_model_name,
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
        token=os.environ.get('HF_TOKEN'),
    )
    
    # Add instruction residuals: θ_final = θ_cpt + Θ_residual
    print("\nApplying instruction residuals to CPT model...")
    cpt_state_dict = cpt_model.state_dict()
    for key in cpt_state_dict.keys():
        if key in instruction_residuals:
            cpt_state_dict[key] = cpt_state_dict[key] + instruction_residuals[key]
            print(f"  Added residual to: {key}")
    
    cpt_model.load_state_dict(cpt_state_dict)
    
    # Save final model
    final_model_name = config['output']['instruct_model']
    print(f"\nSaving final instruct model to: {final_model_name}")
    cpt_model.push_to_hub(final_model_name, token=os.environ.get('HF_TOKEN'))
    tokenizer.push_to_hub(final_model_name, token=os.environ.get('HF_TOKEN'))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Base+CPT model: {base_cpt_model_name}")
    print(f"Instruct model: {final_model_name}")

def main():
    parser = argparse.ArgumentParser(description='Continual Pre-training with Instruction Residuals')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--skip-training', action='store_true', help='Skip CPT, only add residuals')
    parser.add_argument('--cpt-model', type=str, help='Pre-trained CPT model (if skipping training)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Login to HuggingFace
    if 'HF_TOKEN' in os.environ:
        login(token=os.environ['HF_TOKEN'])
    
    # Phase 1: Continual Pre-training
    if args.skip_training:
        if not args.cpt_model:
            raise ValueError("Must provide --cpt-model when using --skip-training")
        base_cpt_model_name = args.cpt_model
        print(f"Skipping training, using existing model: {base_cpt_model_name}")
    else:
        base_cpt_model_name = continual_pretrain(config)
    
    # Phase 2: Add Instruction Residuals
    add_instruction_residuals(config, base_cpt_model_name)

if __name__ == "__main__":
    main()