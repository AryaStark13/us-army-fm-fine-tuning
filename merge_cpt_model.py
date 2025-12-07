"""
This script merges instruction residuals into a CPT model.
Instruction residuals are computed as the difference between
an instruction-tuned model and its base model (parameter-wise).
The final model is obtained by adding these residuals to the CPT model.
"""
import os
import torch
import argparse
from unsloth import FastLanguageModel
from huggingface_hub import login
from dotenv import load_dotenv
import gc

load_dotenv()

def merge_cpt_with_instruction_residuals(base_model_name, instruct_model_name, cpt_model_name, output_model_name, max_seq_length=2048, dtype="bfloat16"):
    """
    Adds instruction residuals to a CPT model using task arithmetic.

    Formula: θ_final = θ_cpt + (θ_instruct - θ_base)

    Args:
        base_model_name: Original base model (e.g., "google/gemma-7b")
        instruct_model_name: Instruction-tuned model (e.g., "google/gemma-7b-it")
        cpt_model_name: CPT model to enhance (e.g., "ShethArihant/gemma-7b-us-army-fm-cpt-base")
        output_model_name: Name for the final model on HuggingFace Hub
        max_seq_length: Maximum sequence length for models
        dtype: Data type for model weights
    """
    print("="*80)
    print("Adding Instruction Residuals to CPT Model")
    print("="*80)
    print(f"\nBase model: {base_model_name}")
    print(f"Instruct model: {instruct_model_name}")
    print(f"CPT model: {cpt_model_name}")
    print(f"Output model: {output_model_name}")

    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")

    # Load original base model on CPU
    print("\n[1/4] Loading original base model on CPU...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=getattr(torch, dtype),
        load_in_4bit=False,
        token=hf_token,
        device_map="cpu",
    )
    base_state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}
    del base_model
    torch.cuda.empty_cache()

    # Load original instruct model on CPU
    print("[2/4] Loading original instruct model on CPU...")
    instruct_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=instruct_model_name,
        max_seq_length=max_seq_length,
        dtype=getattr(torch, dtype),
        load_in_4bit=False,
        token=hf_token,
        device_map="cpu",
    )
    instruct_state_dict = {k: v.cpu() for k, v in instruct_model.state_dict().items()}
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
        model_name=cpt_model_name,
        max_seq_length=max_seq_length,
        dtype=getattr(torch, dtype),
        load_in_4bit=False,
        token=hf_token,
        device_map="cpu",
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
    print(f"\nSaving final model to: {output_model_name}")
    cpt_model.push_to_hub(output_model_name, token=hf_token)
    tokenizer.push_to_hub(output_model_name, token=hf_token)

    print("\n" + "="*80)
    print("MERGE COMPLETE!")
    print("="*80)
    print(f"Final model saved to: {output_model_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Add instruction residuals to a CPT model using task arithmetic"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Original base model (e.g., google/gemma-7b)"
    )
    parser.add_argument(
        "--instruct-model",
        type=str,
        required=True,
        help="Instruction-tuned model (e.g., google/gemma-7b-it)"
    )
    parser.add_argument(
        "--cpt-model",
        type=str,
        required=True,
        help="CPT model to enhance (e.g., ShethArihant/gemma-7b-us-army-fm-cpt-base)"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Name for the final model on HuggingFace Hub"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: bfloat16)"
    )

    args = parser.parse_args()

    # Login to HuggingFace
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)

    merge_cpt_with_instruction_residuals(
        base_model_name=args.base_model,
        instruct_model_name=args.instruct_model,
        cpt_model_name=args.cpt_model,
        output_model_name=args.output_model,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype
    )

if __name__ == "__main__":
    main()
