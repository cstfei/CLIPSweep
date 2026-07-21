python tbarc_unlearning.py \
    --theta_poisoned logs/nodefence_random_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --data_dir data/ImageNet1K/validation \
    --val_set_csv data/ImageNet1K/validation/labels.csv \
    --forget_set_csv data/CC3M/backdoor_abl_backdoor_banana_random_random_16_500000_1500.csv \
    --output_dir logs/tbarc_abl_random \
    --device_id 1 

python tbarc_unlearning.py \
    --theta_poisoned logs/nodefence_blended_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --data_dir data/ImageNet1K/validation \
    --val_set_csv data/ImageNet1K/validation/labels.csv \
    --forget_set_csv data/CC3M/backdoor_abl_backdoor_banana_blended_blended_16_500000_1500.csv \
    --output_dir logs/tbarc_abl_blended \
    --device_id 1 

python tbarc_unlearning.py \
    --theta_poisoned logs/nodefence_SIG_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --data_dir data/ImageNet1K/validation \
    --val_set_csv data/ImageNet1K/validation/labels.csv \
    --forget_set_csv data/CC3M/backdoor_abl_backdoor_banana_SIG_random_16_500000_1500.csv \
    --output_dir logs/tbarc_abl_SIG \
    --device_id 1 

python tbarc_unlearning.py \
    --theta_poisoned logs/nodefence_warped_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --data_dir data/ImageNet1K/validation \
    --val_set_csv data/ImageNet1K/validation/labels.csv \
    --forget_set_csv data/CC3M/backdoor_abl_backdoor_banana_warped_random_16_500000_1500.csv \
    --output_dir logs/tbarc_abl_warped \
    --device_id 1 