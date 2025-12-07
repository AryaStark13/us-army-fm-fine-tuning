"""
Script to merge a PEFT model with its base model and save the merged model.
"""
import os
import gc
import torch
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

load_dotenv()

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()

def merge_and_save_model(
    base_model_name,
    peft_model_path,
    subfolder,
    output_dir,
    hub_model_id=None,
    push_to_hub=False
):
    """
    Load PEFT model, merge with base model, and save
    
    Args:
        base_model_name: HuggingFace model ID or path to base model
        peft_model_path: Path to PEFT adapter checkpoint or HF model ID
        output_dir: Directory to save merged model
        hub_model_id: HuggingFace Hub model ID to push to
        push_to_hub: Whether to push to Hub
    """
    
    # Login to HuggingFace if needed
    hf_token = os.environ.get('HF_TOKEN')
    if push_to_hub and not hf_token:
        raise ValueError("HF_TOKEN required for pushing to Hub")
    if hf_token:
        login(token=hf_token)
    
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading PEFT adapters from: {peft_model_path}")
    
    # Load PEFT model (attaches adapters to base model)
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16
    )
    
    clear_memory()
    
    print("Merging adapters with base model...")
    
    # Merge adapters into base model weights
    model = model.merge_and_unload()
    
    clear_memory()
    
    print(f"Saving merged model to: {output_dir}")
    
    # Save merged model
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Load and save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    
    print(f"Merged model saved successfully to {output_dir}")
    
    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        print(f"Pushing merged model to Hub: {hub_model_id}")
        model.push_to_hub(
            hub_model_id,
            token=hf_token,
            safe_serialization=True,
            commit_message="Merged PEFT model with base model"
        )
        tokenizer.push_to_hub(hub_model_id, token=hf_token)
        print(f"Model successfully pushed to: https://huggingface.co/{hub_model_id}")
    
    clear_memory()
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Merge PEFT model with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="ShethArihant/Llama-3.1-8B-us-army-fm-instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default="./army-fm-sft-final",
        help="Path to PEFT adapter checkpoint or HF model ID"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within PEFT Model ID/Path to load from (eg: checkpoints)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./army-fm-sft-merged",
        help="Directory to save merged model"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="ShethArihant/Llama-3.1-8B-us-army-fm-SFT-instruct",
        help="HuggingFace Hub model ID to push to"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    
    args = parser.parse_args()
    
    merge_and_save_model(
        base_model_name=args.base_model,
        peft_model_path=args.peft_model,
        subfolder=args.subfolder,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub
    )

if __name__ == "__main__":
    main()