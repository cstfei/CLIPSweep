# badnets
python purify_ft_bd_constraint.py \
    --model_path logs/tbarc_abl_random/epoch_5_first_zero_asr_alpha1.00.pt \
    --clean_csv data/CC3M/clean_abl_backdoor_banana_random_random_16_500000_1500.csv \
    --clean_num_samples 5000 \
    --clean_val_ratio 0.02 \
    --clean_seed 42 \
    --bd_val_csv logs/tbarc_abl_random/epoch_5_filtered_forget_set.csv \
    --bd_train_from_val_ratio 0.5 \
    --image_root data/CC3M/ \
    --output_dir logs/bd_constraint_badnets \
    --model_name RN50 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-5 \
    --lambda_init 1.0 \
    --lambda_lr 5.0 \
    --tau_margin 0.02 \
    --device_id 6 \
    --freeze_text \
    --freeze_logit_scale \
    --weight_decay 0.05

# blended
python purify_ft_bd_constraint.py \
    --model_path logs/tbarc_abl_blended/epoch_5_first_zero_asr_alpha0.80.pt \
    --clean_csv data/CC3M/clean_abl_backdoor_banana_blended_blended_16_500000_1500.csv \
    --clean_num_samples 5000 \
    --clean_val_ratio 0.02 \
    --clean_seed 42 \
    --bd_val_csv logs/tbarc_abl_blended/epoch_5_filtered_forget_set.csv \
    --bd_train_from_val_ratio 0.5 \
    --image_root data/CC3M/ \
    --output_dir logs/bd_constraint_blended01 \
    --model_name RN50 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-5 \
    --lambda_init 1.0 \
    --lambda_lr 5.0 \
    --tau_margin 0.02 \
    --device_id 4 \
    --freeze_text \
    --freeze_logit_scale \
    --weight_decay 0.01

# SIG
python purify_ft_bd_constraint.py \
    --model_path logs/tbarc_abl_SIG/epoch_5_first_zero_asr_alpha0.80.pt \
    --clean_csv data/CC3M/clean_abl_backdoor_banana_SIG_random_16_500000_1500.csv \
    --clean_num_samples 5000 \
    --clean_val_ratio 0.02 \
    --clean_seed 42 \
    --bd_val_csv logs/tbarc_abl_SIG/epoch_5_filtered_forget_set.csv \
    --bd_train_from_val_ratio 0.5 \
    --image_root data/CC3M/ \
    --output_dir logs/bd_constraint_SIG01 \
    --model_name RN50 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-5 \
    --lambda_init 1.0 \
    --lambda_lr 5.0 \
    --tau_margin 0.02 \
    --device_id 6 \
    --freeze_text \
    --freeze_logit_scale \
    --weight_decay 0.01

# warped
python purify_ft_bd_constraint.py \
    --model_path logs/tbarc_abl_warped/epoch_5_first_zero_asr_alpha1.00.pt \
    --clean_csv data/CC3M/clean_abl_backdoor_banana_warped_random_16_500000_1500.csv \
    --clean_num_samples 5000 \
    --clean_val_ratio 0.02 \
    --clean_seed 42 \
    --bd_val_csv logs/tbarc_abl_warped/epoch_5_filtered_forget_set.csv \
    --bd_train_from_val_ratio 0.5 \
    --image_root data/CC3M/ \
    --output_dir logs/bd_constraint_warped01 \
    --model_name RN50 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-5 \
    --lambda_init 1.0 \
    --lambda_lr 5.0 \
    --tau_margin 0.02 \
    --device_id 4 \
    --freeze_text \
    --freeze_logit_scale \
    --weight_decay 0.01