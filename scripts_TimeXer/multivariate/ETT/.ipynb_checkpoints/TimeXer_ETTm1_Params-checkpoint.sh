export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $1 \
  --data ETTm1 \
  --features M \
  --seq_len $3 \
  --label_len 48 \
  --pred_len $4 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads $5 \
  --d_model $6 \
  --d_ff $7 \
  --batch_size 32 \
  --des 'Exp' \
  --itr $2 \
  --topk_ratio 0.1 \
  --dropout_alpha 0.1 \
  --dropout_data 0.1