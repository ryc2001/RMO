### RMO: Towards Better LLM Alignment via Reshaping Reward Margin Distributions

- Our repo is based on the LLaMA-Factory. Please first download the repository and install the corresponding environment.
- Next, please move the `data` folder and other files from our code into LLaMA-Factory.

### Dual Denoising Filtering

- We provide a running example using  `Llama-3.2-3b-Instruct ` as the model:

``python dual_filter.py --file_before logp_margin_llama_3b_instruct.txt --file_after logp_margin_llama_3b.txt --output llama_3b_filter_indices.txt``

### Batch Margin Diversification

- We provide a running example using  `Llama-3.2-3b-Instruct ` as the model:

``python get_high_batch.py \ --margins logp_margin_llama_3b.txt \ --partition_out llama_3b_batches.json \ --train_in data/hHH_opt/train.jsonl \ --train_out data/HH_opt_llama_3b_high/train.jsonl \ --batch_size 16 \ --restarts 200 \ --iterations 100000 \ --show_inner ``

### Supervised Finetuning


- We provide a running example using  `pythia-2.8b ` as the model:

``CUDA_VISIBLE_DEVICES=0 python src/train.py   --stage sft   --model_name_or_path EleutherAI/pythia-2.8b   --do_train   --dataset_dir ./data   --dataset HH-sft   --template chatml   --finetuning_type lora   --lora_target all   --output_dir ./saves/pythia-2.8b-sft_lora   --overwrite_cache   --overwrite_output_dir   --cutoff_len 2048   --learning_rate 5e-5   --num_train_epochs 3   --per_device_train_batch_size 16   --gradient_accumulation_steps 1   --logging_steps 10   --save_steps 100   --fp16   --resize_vocab True ``

### Preference Optimization

- We provide a running example using  `pythia-2.8b ` as the model:

``python src/train.py --do\_train --model\_name\_or\_path `./saves/pythia-2.8b-sft` --stage dpo --num\_train\_epochs 3 --dataset HH-opt --template default --finetuning\_type lora --output\_dir saves/pythia-2.8b-opt\_lora --per\_device\_train\_batch\_size 16 --ignor\_data\_skip --gradient\_accumulation\_steps 1 --overwrite\_output\_dir --logging\_steps 1 --report\_to none --disable\_shuffling True --gradient\_checkpointing --lambda\_reg 0   --margin\_target 1.82   --margin\_scaling 125.0``

In this example, the three parameters `--lambda_reg`, `--margin_target`, and `--margin_scaling` are hyperparameters used in the RMO  process.

