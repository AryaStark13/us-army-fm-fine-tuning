python merge_peft_model.py \
--peft_model ShethArihant/army-fm-sft-checkpoints/checkpoint-1400 \
--output_dir army-fm-sft-checkpoints/checkpoint-1400

python merge_peft_model.py \
--peft_model ShethArihant/army-fm-sft-checkpoints/checkpoint-2200 \
--output_dir army-fm-sft-checkpoints/checkpoint-2200

python merge_peft_model.py \
--peft_model ShethArihant/army-fm-sft-checkpoints/checkpoint-2379 \
--output_dir army-fm-sft-checkpoints/checkpoint-2379

# hf upload ShethArihant/Llama-3.1-8B-us-army-fm-SFT-instruct-checkpoints army-fm-sft-checkpoints/
hf upload-large-folder ShethArihant/Llama-3.1-8B-us-army-fm-SFT-instruct-checkpoints --repo-type=model army-fm-sft-checkpoints/ --num-workers=16