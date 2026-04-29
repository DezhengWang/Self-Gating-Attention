export CUDA_VISIBLE_DEVICES="0,1,2,3"
model=PAttn
methods_m='PAttn'

percent=100
patience=10
tag_file=main.py

inp_len_m=512
pre_lens_m="96 192 336 720"
filename_m=ETTm_simple.txt
lr=0.00005
gpu_loc=0
itt=1
for pred_len in $pre_lens_m;
do
for method in $methods_m;
do
python $tag_file \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model $MODEL \
    --data custom \
    --seq_len $inp_len_m \
    --label_len 48 \
    --pred_len $pred_len \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 6 \
    --gpu_loc $gpu_loc \
    --patience $patience \
    --method $method \
    --model $model \
    --cos 1 \
    --is_gpt 1 \
    --itr $itt \
done
done
