export CUDA_VISIBLE_DEVICES=0

model_name=$1

for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr $2 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001
    
done


for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr $2 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001

done
