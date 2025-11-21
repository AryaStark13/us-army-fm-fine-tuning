import os
import yaml
import torch
from unsloth import FastLanguageModel
from huggingface_hub import login
import argparse

from dotenv import load_dotenv
load_dotenv()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def push_cpt_model(config, checkpoint_path):
    """Load checkpoint, merge LoRA, and push base+CPT model"""
    print("="*80)
    print("STEP 1: Pushing Base+CPT Model")
    print("="*80)
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint (which has LoRA adapters)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    print("Merging LoRA weights into base model...")
    model = FastLanguageModel.for_inference(model)
    merged_model = model.merge_and_unload()
    
    # Push to HuggingFace
    base_cpt_name = config['output']['base_cpt_model']
    print(f"\nPushing base+CPT model to: {base_cpt_name}")
    merged_model.push_to_hub(base_cpt_name, token=os.environ.get('HF_TOKEN'))
    tokenizer.push_to_hub(base_cpt_name, token=os.environ.get('HF_TOKEN'))
    
    print(f"✅ Successfully pushed: {base_cpt_name}")
    return base_cpt_name

def push_instruct_model(config, base_cpt_model_name):
    """Add instruction residuals and push instruct model"""
    print("\n" + "="*80)
    print("STEP 2: Adding Instruction Residuals and Pushing Instruct Model")
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
    
    # Compute instruction residuals
    print("[3/4] Computing instruction residuals...")
    instruction_residuals = {}
    residual_count = 0
    for key in base_state_dict.keys():
        if key in instruct_state_dict:
            instruction_residuals[key] = instruct_state_dict[key] - base_state_dict[key]
            residual_count += 1
    
    print(f"  Computed {residual_count} instruction residuals")
    
    del base_state_dict, instruct_state_dict
    torch.cuda.empty_cache()
    
    # Load CPT model from HuggingFace
    print("[4/4] Loading CPT model from HuggingFace and adding residuals...")
    cpt_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_cpt_model_name,
        max_seq_length=config['training']['max_seq_length'],
        dtype=getattr(torch, config['model']['dtype']),
        load_in_4bit=config['model']['load_in_4bit'],
        token=os.environ.get('HF_TOKEN'),
    )
    
    # Add instruction residuals
    print("\nApplying instruction residuals to CPT model...")
    cpt_state_dict = cpt_model.state_dict()
    applied_count = 0
    for key in cpt_state_dict.keys():
        if key in instruction_residuals:
            cpt_state_dict[key] = cpt_state_dict[key] + instruction_residuals[key]
            applied_count += 1
    
    print(f"  Applied {applied_count} instruction residuals")
    
    cpt_model.load_state_dict(cpt_state_dict)
    
    # Push final model
    final_model_name = config['output']['instruct_model']
    print(f"\nPushing instruct model to: {final_model_name}")
    cpt_model.push_to_hub(final_model_name, token=os.environ.get('HF_TOKEN'))
    tokenizer.push_to_hub(final_model_name, token=os.environ.get('HF_TOKEN'))
    
    print(f"✅ Successfully pushed: {final_model_name}")

def main():
    parser = argparse.ArgumentParser(description='Push models to HuggingFace Hub')
    parser.add_argument('--config', type=str, default='cpt_training_config.yaml', 
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint-504',
                        help='Path to checkpoint directory')
    parser.add_argument('--skip-cpt', action='store_true',
                        help='Skip pushing base+CPT model (if already pushed)')
    parser.add_argument('--cpt-model', type=str,
                        help='HF model name for base+CPT (if skipping CPT push)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Login to HuggingFace
    if 'HF_TOKEN' in os.environ:
        login(token=os.environ['HF_TOKEN'])
    else:
        raise ValueError("HF_TOKEN environment variable not set!")
    
    # Step 1: Push base+CPT model
    if args.skip_cpt:
        if not args.cpt_model:
            base_cpt_model_name = config['output']['base_cpt_model']
            print(f"Skipping CPT push, using: {base_cpt_model_name}")
        else:
            base_cpt_model_name = args.cpt_model
            print(f"Skipping CPT push, using: {base_cpt_model_name}")
    else:
        base_cpt_model_name = push_cpt_model(config, args.checkpoint)
    
    # Step 2: Add residuals and push instruct model
    # push_instruct_model(config, base_cpt_model_name)
    
    print("\n" + "="*80)
    print("✅ ALL MODELS PUSHED SUCCESSFULLY!")
    print("="*80)
    print(f"Base+CPT model: {base_cpt_model_name}")
    print(f"Instruct model: {config['output']['instruct_model']}")

if __name__ == "__main__":
    main()