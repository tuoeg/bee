python3 -m torch.distributed.launch \
  --nproc_per_node=1  unilm/layoutlmv3/examples/run_funsd_cord.py \
  --dataset_name funsd \
  --do_train --do_eval \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir test \
  --segment_level_layout 1 --visual_embed 1 --input_size 224 \
  --max_steps 1000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
  --learning_rate 1e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --dataloader_num_workers 8

mv *.npy ./data
rm ./test -r
