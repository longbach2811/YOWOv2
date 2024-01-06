python train.py \
        -d custom \
        -v yowo_v2_large \
        --root /home/longbach/Desktop/motion-det-dataset/processed_data_v3 \
        --num_workers 4 \
        --eval_epoch 1 \
        --max_epoch 7 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 8 \
        -accu 16 \
        -K 5
        # --eval \
