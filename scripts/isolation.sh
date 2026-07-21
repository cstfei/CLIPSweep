# !/bin/bash

# badnet
python -m backdoor_isolation \
    --name def_clean_badnet \
    --train_data data/CC3M/backdoor_banana_random_random_16_500000_1500.csv \
    --checkpoint logs/nodefence_random_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --tuning_epochs 10 \
    --isolation_ratio 0.1 \
    --batch_size 512 \
    --num_workers 16 \
    --gamma 0.5 \
    --device_id 7 \
    --eval_batch_size_multiplier 3

# blended
python -m backdoor_isolation \
    --name def_clean_blended \
    --train_data data/CC3M/backdoor_banana_blended_blended_16_500000_1500.csv \
    --checkpoint logs/nodefence_blended_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --tuning_epochs 10 \
    --isolation_ratio 0.1 \
    --batch_size 512 \
    --num_workers 16 \
    --gamma 0.5 \
    --device_id 7 \
    --eval_batch_size_multiplier 3

# SIG
python -m backdoor_isolation \
    --name def_clean_sig \
    --train_data data/CC3M/backdoor_banana_SIG_random_16_500000_1500.csv \
    --checkpoint logs/nodefence_SIG_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --tuning_epochs 10 \
    --isolation_ratio 0.004 \
    --batch_size 512 \
    --num_workers 16 \
    --gamma 0.5 \
    --device_id 7 \
    --eval_batch_size_multiplier 3

# warped
python -m backdoor_isolation \
    --name def_clean_warped \
    --train_data data/CC3M/backdoor_banana_warped_random_16_500000_1500.csv \
    --checkpoint logs/nodefence_warped_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --tuning_epochs 10 \
    --isolation_ratio 0.004 \
    --batch_size 512 \
    --num_workers 16 \
    --gamma 0.5 \
    --device_id 7 \
    --eval_batch_size_multiplier 3