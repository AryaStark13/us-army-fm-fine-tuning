import os
import gc
import torch
import wandb
import yaml
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

load_dotenv()

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def add_eos_to_assistant_messages(example, eos_token):
    """Add EOS token to the end of each assistant message"""
    messages = example['messages']
    for message in messages:
        if message['role'] == 'assistant':
            message['content'] = message['content'] + eos_token
    return {'messages': messages}

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Login to HuggingFace and WandB
    hf_token = os.environ.get('HF_TOKEN')
    wandb_key = os.environ.get('WANDB_API_KEY')

    # Check if tokens are provided
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables.")
        raise ValueError("HuggingFace token (HF_TOKEN) is required.")
    if not wandb_key:
        print("Warning: WANDB_API_KEY not found in environment variables.")
        raise ValueError("WandB API key (WANDB_API_KEY) is required.")

    if hf_token:
        login(token=hf_token)
    
    if wandb_key:
        wandb.login(key=wandb_key)
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['run_name'],
        config=config
    )
    
    print("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['tokenizer_name'],
        trust_remote_code=True
    )

    # Set pad token if it doesn't exist (e.g., for LLaMA)
    # Models like Gemma already have a dedicated pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Pad token not found. Setting pad_token = eos_token ({tokenizer.eos_token})")
    else:
        print(f"Using existing pad_token: {tokenizer.pad_token}")

    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM",
        target_modules=config['lora']['target_modules']
    )
    
    print("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    clear_memory()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config['data']['dataset_name'])

    # Add EOS token to assistant messages
    print("Adding EOS token to assistant messages...")
    train_dataset = dataset['train'].map(
        lambda x: add_eos_to_assistant_messages(x, tokenizer.eos_token),
        desc="Adding EOS to train dataset"
    )
    eval_dataset = dataset['test'].map(
        lambda x: add_eos_to_assistant_messages(x, tokenizer.eos_token),
        desc="Adding EOS to eval dataset"
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        eval_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=True,
        optim=config['training']['optimizer'],
        max_grad_norm=config['training']['max_grad_norm'],
        report_to="wandb",
        run_name=config['wandb']['run_name'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=config['training']['max_length'],
        packing=False,
        dataset_text_field="messages",
        dataset_kwargs={"skip_prepare_dataset": False}
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    clear_memory()
    
    # Train
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(config['training']['final_model_dir'])
    
    clear_memory()
    
    # Push to Hub
    if config['hub']['push_to_hub']:
        print("Merging LoRA adapters with base model...")
        merged_model = trainer.model.merge_and_unload()

        print(f"Pushing merged model to Hub: {config['hub']['hub_model_id']}")
        merged_model.push_to_hub(
            config['hub']['hub_model_id'],
            token=hf_token,
            commit_message="SFT fine-tuned model (merged)"
        )
        tokenizer.push_to_hub(
            config['hub']['hub_model_id'],
            token=hf_token
        )
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="sft_training_config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config)