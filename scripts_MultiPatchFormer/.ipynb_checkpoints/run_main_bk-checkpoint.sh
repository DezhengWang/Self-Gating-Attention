#!/bin/bash
# Set the GPU device ID to use (only use GPU 1, index starts from 0)
export CUDA_VISIBLE_DEVICES=0

# Check if the model name parameter is provided
# $1 represents the first argument passed to the script (model name)
if [ -z "$1" ]; then
    echo "Error: Please provide the model name as the first parameter!"
    echo "Usage: bash $0 <model_name>"
    exit 1
fi

# Define parameter lists for iteration
ITRS=(0 1 2)          # List of iteration numbers for experiments
PRELENS=(96 192 336 720)  # List of prediction lengths to test
MODEL=$1              # Model name passed from command line

# 定义run.py的绝对路径（核心修正）
RUN_PY_PATH="/mnt/sdb1/proj/20_SGA/run.py"

# 验证run.py是否存在
if [ ! -f "$RUN_PY_PATH" ]; then
    echo "Error: run.py not found at $RUN_PY_PATH"
    exit 1
fi

# Create log directory to store running logs (optional)
LOG_DIR="./run_logs_${MODEL}"
mkdir -p $LOG_DIR

# Double loop: iterate over all combinations of itr and preLen
for itr in "${ITRS[@]}"; do
    for preLen in "${PRELENS[@]}"; do
        echo "========================================"
        echo "Starting run: model=$MODEL, itr=$itr, preLen=$preLen"
        echo "Current time: $(date)"
        echo "========================================"

        # Define log file path for each parameter combination
        LOG_FILE="${LOG_DIR}/${MODEL}_itr${itr}_preLen${preLen}.log"

        # Run experiment on ETTm1 dataset
        echo "Running ETTm1 dataset..."
        python -u "$RUN_PY_PATH" \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm1.csv \
          --model_id ETTm1 \
          --model $MODEL \
          --data ETTm1 \
          --features M \
          --learning_rate 0.0001\
          --train_epochs 10 \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $preLen \
          --e_layers 1 \
          --d_layers 1 \
          --d_model 256 \
          --n_heads 8 \
          --d_ff 512 \
          --dropout 0.1 \
          --patch_len 16 \
          --stride 16 \
          --factor 3 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --itr $itr >> $LOG_FILE 2>&1

        # Run experiment on ETTh1 dataset
        echo "Running ETTh1 dataset..."
        python -u "$RUN_PY_PATH" \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh1.csv \
          --model_id ETTh1 \
          --model $MODEL \
          --data ETTh1 \
          --features M \
          --learning_rate 0.0001\
          --train_epochs 10 \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $preLen \
          --e_layers 1 \
          --d_layers 1 \
          --d_model 256 \
          --n_heads 8 \
          --d_ff 256 \
          --dropout 0.1 \
          --patch_len 16 \
          --stride 8 \
          --factor 3 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --itr $itr >> $LOG_FILE 2>&1

        # Run experiment on ETTm2 dataset
        echo "Running ETTm2 dataset..."
        python -u "$RUN_PY_PATH" \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm2.csv \
          --model_id ETTm2 \
          --model $MODEL \
          --data ETTm2 \
          --features M \
          --learning_rate 0.0001\
          --train_epochs 10 \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $preLen \
          --e_layers 1 \
          --d_layers 1 \
          --d_model 256 \
          --n_heads 8 \
          --d_ff 512 \
          --dropout 0.1 \
          --patch_len 16 \
          --stride 16 \
          --factor 3 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --itr $itr >> $LOG_FILE 2>&1

        # Run experiment on ETTh2 dataset
        echo "Running ETTh2 dataset..."
        python -u "$RUN_PY_PATH" \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh2.csv \
          --model_id ETTh2 \
          --model $MODEL \
          --data ETTh2 \
          --features M \
          --learning_rate 0.0001\
          --train_epochs 10 \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $preLen \
          --e_layers 1 \
          --d_layers 1 \
          --d_model 256 \
          --n_heads 8 \
          --d_ff 256 \
          --dropout 0.1 \
          --patch_len 16 \
          --stride 8 \
          --factor 3 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --itr $itr >> $LOG_FILE 2>&1

        # Run experiment on exchange rate dataset
        echo "Running exchange dataset..."
        python -u "$RUN_PY_PATH" \
         --is_training 1 \
         --root_path ./dataset/exchange_rate/ \
         --data_path exchange_rate.csv \
         --model_id Exchange \
         --model $MODEL \
         --data custom \
         --features M \
         --learning_rate 0.0001\
         --train_epochs 10 \
         --seq_len 96 \
         --label_len 48 \
         --pred_len $preLen \
         --e_layers 1 \
         --d_layers 1 \
         --d_model 256 \
         --n_heads 8 \
         --d_ff 512 \
         --dropout 0.1 \
         --enc_in 8 \
         --dec_in 8 \
         --c_out 8 \
         --freq 0 \
         --patch_len 16 \
         --stride 16 \
         --factor 3 \
         --des 'Exp' \
         --itr $itr >> $LOG_FILE 2>&1

        # Run experiment on weather dataset
        echo "Running weather dataset..."
        python -u "$RUN_PY_PATH" \
         --is_training 1 \
         --root_path ./dataset/weather/ \
         --data_path weather.csv \
         --model_id weather \
         --model $MODEL \
         --data custom \
         --features M \
         --learning_rate 0.0001\
         --train_epochs 10 \
         --seq_len 96 \
         --label_len 48 \
         --pred_len $preLen \
         --e_layers 1 \
         --d_layers 1 \
         --d_model 256 \
         --n_heads 8 \
         --d_ff 512 \
         --dropout 0.1 \
         --patch_len 16 \
         --stride 8 \
         --factor 3 \
         --enc_in 21 \
         --dec_in 21 \
         --c_out 21 \
         --des 'Exp' \
         --itr $itr >> $LOG_FILE 2>&1

        echo "Completed: model=$MODEL, itr=$itr, preLen=$preLen"
        echo "Log file: $LOG_FILE"
        echo ""
    done
done

echo "All combinations have been executed successfully!"
echo "Log files are saved in ${LOG_DIR}/ directory"
