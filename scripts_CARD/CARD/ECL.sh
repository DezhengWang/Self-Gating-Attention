export CUDA_VISIBLE_DEVICES=0

model_name=$1

for pred_len in 96 192 336 720
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des Exp \
      --itr 1 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --patch_len 16\
      --stride 8\
      --patience 100\
      --train_epochs 100 --lradj CARD \
      --itr $2 --batch_size 32 --learning_rate 0.0001 \
      --dp_rank 8 --warmup_epochs 20
    
done