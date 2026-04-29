export CUDA_VISIBLE_DEVICES=1

#cd ..

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather \
  --model $1 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --itr $3 \
  --topk_ratio 0.9 \
  --dropout_alpha 0.5 \
  --dropout_data 0.9
