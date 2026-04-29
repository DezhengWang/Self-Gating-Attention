export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $1 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $3 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --batch_size 4 \
  --des 'exp' \
  --itr $2 \
  --topk_ratio 0.1 \
  --dropout_alpha 0.1 \
  --dropout_data 0.1
