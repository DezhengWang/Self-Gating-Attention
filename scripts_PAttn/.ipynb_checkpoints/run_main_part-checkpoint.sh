#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


# Run experiment on Weather dataset
echo "Running Weather dataset..."

python -u run.py \
  --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --model_id weather \
 --model $1 \
 --data custom \
 --features M \
 --learning_rate 0.00005\
 --train_epochs 10 \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $2 \
 --e_layers 3 \
 --d_layers 1 \
 --d_model 768 \
 --n_heads 4 \
 --d_ff 768 \
 --dropout 0.3 \
 --patch_len 16 \
 --stride 8 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
  --des 'Exp' \
  --itr $3 \
  --gpu 0 \
  --topk_ratio 0.5\
  --dropout_alpha 0.3 \
  --dropout_data 0.1