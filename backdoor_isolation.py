'''
ABL-style Backdoor Isolation for CLIP
Usage:
python -m backdoor_isolation \
    --name def_clean \
    --train_data data/GCC-training/backdoor_banana_blended_blended_16_500000_1500_train.csv \
    --checkpoint logs/nodefence_random_500000_1500_bs128_1e-6_ep5/checkpoints/epoch_5.pt \
    --device_id 2 \
    --tuning_epochs 5 \
    --isolation_ratio 0.004 \
    --gamma 0.5
'''
import os
from tqdm import tqdm
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from src.train import get_loss
import pandas as pd
import torch
import logging
import warnings
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import clip
from pkgs.openai.clip import load as load_model
from src.parser import parse_args
from src.logger import get_logger, set_logger
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset, DataLoader

PngImagePlugin.MAX_TEXT_CHUNK = 50 * 1024 * 1024

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

class ImageCaptionDataset(Dataset):
    """Dataset for CLIP image-text pairs"""
    def __init__(self, data, root, preprocess):
        self.data = data
        self.root = root
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image"]
        caption = row["caption"]
        
        try:
            image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
            image = self.preprocess.process_image(image)
        except Exception as e:
            logging.warning(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = torch.zeros(3, 224, 224)
        
        return image, caption, image_path, idx


def gradient_ascent_step(model, images, texts, preprocess, device, gamma=0.5, gradient_type='LGA'):
    """
    Perform one gradient ascent step (ABL-style)
    LGA (Local Gradient Ascent): loss_ascent = sign(loss - gamma) * loss
    """
    # 获取底层模型（处理 DDP 包装）
    base_model = model.module if isinstance(model, DDP) else model
    
    model.train()
    
    # Tokenize texts
    text_tokens = preprocess.process_text(texts)
    text_input_ids = text_tokens["input_ids"].to(device)
    text_attention_mask = text_tokens["attention_mask"].to(device)
    
    # Forward pass
    image_features = base_model.get_image_features(images)
    text_features = base_model.get_text_features(
        input_ids=text_input_ids,
        attention_mask=text_attention_mask
    )
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # CLIP loss (contrastive)
    logit_scale = base_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2
    
    # Apply gradient ascent transformation
    if gradient_type == 'LGA':
        # Local Gradient Ascent
        loss_ascent = torch.sign(loss - gamma) * loss
    elif gradient_type == 'Flooding':
        # Flooding (alternative method)
        loss_ascent = (loss - gamma).abs() + gamma
    else:
        loss_ascent = loss
    
    return loss_ascent, loss.item()


def train_ascent_model(model, dataloader, preprocess, device, options, is_distributed=False):
    """
    Train model with gradient ascent to amplify backdoor behavior
    Supports multi-GPU distributed training
    """
    if options.master or not is_distributed:
        logging.info("=" * 60)
        logging.info("Stage 1: Gradient Ascent Training (Amplifying backdoor signals)")
        if is_distributed:
            logging.info(f"Using {options.num_devices} GPUs for distributed training")
        logging.info("=" * 60)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=options.lr if hasattr(options, 'lr') else 0.001,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 使用混合精度训练
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    tuning_epochs = options.tuning_epochs if hasattr(options, 'tuning_epochs') else 5
    gamma = options.gamma if hasattr(options, 'gamma') else 0.5
    
    if options.master or not is_distributed:
        logging.info(f"Using AMP: {use_amp}")
    
    model.train()
    
    for epoch in range(tuning_epochs):
        # 分布式训练时设置epoch以确保每个epoch的shuffle不同
        if is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0
        num_batches = 0
        
        # 只在主进程显示进度条
        if options.master or not is_distributed:
            pbar = tqdm(dataloader, desc=f"Ascent Epoch {epoch+1}/{tuning_epochs}")
        else:
            pbar = dataloader
        
        for images, captions, _, _ in pbar:
            images = images.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用混合精度
            with autocast(enabled=use_amp):
                # Gradient ascent step
                loss_ascent, loss_value = gradient_ascent_step(
                    model, images, captions, preprocess, device, gamma=gamma
                )
            
            if use_amp:
                scaler.scale(loss_ascent).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_ascent.backward()
                optimizer.step()
            
            epoch_loss += loss_value
            num_batches += 1
            
            if options.master or not is_distributed:
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': f'{loss_value:.4f}'})
        
        # 同步所有进程的loss统计
        if is_distributed:
            epoch_loss_tensor = torch.tensor([epoch_loss, num_batches], dtype=torch.float32).to(device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor[0].item()
            num_batches = int(epoch_loss_tensor[1].item())
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if options.master or not is_distributed:
            logging.info(f"Epoch {epoch+1}/{tuning_epochs} - Average Loss: {avg_loss:.4f}")
    
    if options.master or not is_distributed:
        logging.info("Gradient ascent training completed")
    return model


def compute_loss_per_sample(model, dataloader, preprocess, device, options, is_distributed=False):
    """
    Compute loss value for each sample (ABL-style)
    Lower loss = likely backdoor sample
    Optimized: batch computation instead of per-sample
    Supports distributed inference with result gathering
    """
    if options.master or not is_distributed:
        logging.info("=" * 60)
        logging.info("Stage 2: Computing loss for each sample (Batch mode)")
        logging.info("=" * 60)
    
    # 如果是DDP模型，使用底层模型进行评估
    base_model = model.module if isinstance(model, DDP) else model
    
    base_model.eval()
    
    loss_records = []
    image_paths_all = []
    captions_all = []
    original_indices = []  # 记录原始索引
    
    with torch.no_grad():
        if options.master or not is_distributed:
            pbar = tqdm(dataloader, desc="Computing losses")
        else:
            pbar = dataloader
        
        for images, caps, img_paths, indices in pbar:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            
            # Tokenize
            text_tokens = preprocess.process_text(caps)
            text_input_ids = text_tokens["input_ids"].to(device)
            text_attention_mask = text_tokens["attention_mask"].to(device)
            
            # Forward (batch computation)
            image_features = base_model.get_image_features(images)
            text_features = base_model.get_text_features(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # CLIP loss for all samples in batch (vectorized)
            logit_scale = base_model.logit_scale.exp()
            
            # Compute cosine similarity for each image-text pair
            # For i-th sample, we want similarity between image[i] and text[i]
            similarities = logit_scale * torch.sum(image_features * text_features, dim=-1)
            
            # For contrastive learning, the correct pair should have highest similarity
            # We compute full similarity matrix for proper contrastive loss
            logits_matrix = logit_scale * image_features @ text_features.t()
            labels = torch.arange(batch_size).to(device)
            
            # Compute loss for each sample in batch
            losses_batch = nn.CrossEntropyLoss(reduction='none')(logits_matrix, labels)
            
            # Store results with original indices
            loss_records.extend(losses_batch.cpu().numpy().tolist())
            image_paths_all.extend(img_paths)
            captions_all.extend(caps)
            original_indices.extend(indices.cpu().numpy().tolist())  # 保存原始索引
    
    # 分布式训练时收集所有进程的结果
    if is_distributed:
        # 使用all_gather收集所有进程的结果
        all_loss_records = [None for _ in range(options.num_devices)]
        all_image_paths = [None for _ in range(options.num_devices)]
        all_captions = [None for _ in range(options.num_devices)]
        all_original_indices = [None for _ in range(options.num_devices)]
        
        dist.all_gather_object(all_loss_records, loss_records)
        dist.all_gather_object(all_image_paths, image_paths_all)
        dist.all_gather_object(all_captions, captions_all)
        dist.all_gather_object(all_original_indices, original_indices)
        
        # 展平列表
        loss_records = [item for sublist in all_loss_records for item in sublist]
        image_paths_all = [item for sublist in all_image_paths for item in sublist]
        captions_all = [item for sublist in all_captions for item in sublist]
        original_indices = [item for sublist in all_original_indices for item in sublist]
        
        # 去除DistributedSampler可能产生的重复索引
        # 创建索引到数据的映射，保留每个索引的第一次出现
        seen_indices = {}
        unique_loss_records = []
        unique_image_paths = []
        unique_captions = []
        unique_indices = []
        
        for i, idx in enumerate(original_indices):
            if idx not in seen_indices:
                seen_indices[idx] = True
                unique_loss_records.append(loss_records[i])
                unique_image_paths.append(image_paths_all[i])
                unique_captions.append(captions_all[i])
                unique_indices.append(idx)
        
        loss_records = unique_loss_records
        image_paths_all = unique_image_paths
        captions_all = unique_captions
        original_indices = unique_indices
    
    # Sort by loss (ascending), but keep track of original indices
    sorted_positions = np.argsort(np.array(loss_records))
    loss_indices = [original_indices[i] for i in sorted_positions]  # 使用原始索引
    
    if options.master or not is_distributed:
        logging.info(f"Loss statistics:")
        logging.info(f"  Total samples: {len(loss_records)}")
        logging.info(f"  Min loss: {np.min(loss_records):.4f}")
        logging.info(f"  Max loss: {np.max(loss_records):.4f}")
        logging.info(f"  Mean loss: {np.mean(loss_records):.4f}")
        logging.info(f"  Median loss: {np.median(loss_records):.4f}")
        logging.info(f"  Top 10 lowest losses: {np.sort(np.array(loss_records))[:10]}")
    
    return loss_records, loss_indices, image_paths_all, captions_all


def worker(rank, options, logger):
    """
    Main worker function implementing ABL-style backdoor isolation
    Supports single-GPU and multi-GPU distributed training
    """
    options.rank = rank
    options.master = rank == 0

    set_logger(rank=rank, logger=logger, distributed=options.distributed)
    
    # 设置日志级别（非主进程减少输出）
    if options.distributed and not options.master:
        logging.getLogger().setLevel(logging.WARNING)
    
    if options.device == "cuda":
        device_id = options.device_ids[rank] if options.distributed else options.device_id
        device = f"cuda:{device_id}"
        options.device = device
        torch.cuda.set_device(device_id)
    else:
        device = options.device
    
    # 初始化分布式进程组
    if options.distributed:
        dist.init_process_group(
            backend=options.distributed_backend,
            init_method=options.distributed_init_method,
            world_size=options.num_devices,
            rank=options.rank
        )
    
    # Load poisoned model
    if options.master:
        logging.info("Loading poisoned CLIP model...")
    model, preprocess = load_model(name=options.model_name, pretrained=False)
    model.to(device)
    
    # Load checkpoint (must be poisoned model)
    if options.checkpoint is not None:
        if os.path.isfile(options.checkpoint):
            if options.master:
                logging.info(f"Loading checkpoint from {options.checkpoint}")
            
            checkpoint = torch.load(options.checkpoint, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                raise ValueError("Unknown checkpoint format")
            
            # Handle module prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'module.' prefix if present
                new_key = key[len("module."):] if key.startswith("module.") else key
                new_state_dict[new_key] = value
            
            # Load state dict with error handling
            try:
                model.load_state_dict(new_state_dict, strict=True)
                if options.master:
                    logging.info(f"Loaded poisoned model successfully (strict mode)")
            except RuntimeError as e:
                if options.master:
                    logging.warning(f"Strict loading failed: {e}")
                    logging.info("Trying non-strict loading...")
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if options.master:
                    if missing:
                        logging.warning(f"Missing keys: {missing[:5]}...")
                    if unexpected:
                        logging.warning(f"Unexpected keys: {unexpected[:5]}...")
                    logging.info("Loaded poisoned model successfully (non-strict mode)")
        else:
            raise FileNotFoundError(f"No checkpoint found at {options.checkpoint}")
    else:
        raise ValueError("Must provide --checkpoint with poisoned model path")
    
    # 分布式训练时同步，确保所有进程都加载完成
    if options.distributed:
        dist.barrier()
    
    # 包装为DDP模型（分布式训练）
    if options.distributed:
        model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
        if options.master:
            logging.info(f"Model wrapped with DDP on GPU {device_id}")
    
    # Load data
    csv_file = options.train_data
    root = os.path.dirname(csv_file)
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    data = pd.read_csv(csv_file)
    
    logging.info(f"Loaded {len(data)} samples from {csv_file}")
    
    # Create dataset and dataloader
    dataset = ImageCaptionDataset(data, root, preprocess)
    
    # 优化DataLoader配置以提升GPU利用率
    # 分布式训练时，每个GPU的workers数量
    num_workers = min(options.num_workers // options.num_devices if options.distributed else options.num_workers, 16)
    
    # 分布式训练时使用DistributedSampler
    train_sampler = None
    if options.distributed:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=options.num_devices,
            rank=options.rank,
            shuffle=True
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=(train_sampler is None),  # 使用sampler时不能shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,  # 分布式训练时禁用pin_memory避免CUDA错误
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    # Create evaluation dataloader (no shuffle, larger batch for faster inference)
    multiplier = options.eval_batch_size_multiplier if hasattr(options, 'eval_batch_size_multiplier') else 2
    eval_batch_size = min(options.batch_size * multiplier, 512)
    
    # 分布式推理时也使用DistributedSampler（但不shuffle）
    eval_sampler = None
    if options.distributed:
        eval_sampler = DistributedSampler(
            dataset,
            num_replicas=options.num_devices,
            rank=options.rank,
            shuffle=False
        )
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=num_workers,
        pin_memory=False,  # 分布式训练时禁用pin_memory避免CUDA错误
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    if options.master:
        logging.info(f"Training batch size per GPU: {options.batch_size}")
        logging.info(f"Evaluation batch size per GPU: {eval_batch_size}")
        logging.info(f"Number of workers per GPU: {num_workers}")
        if options.distributed:
            logging.info(f"Total effective batch size: {options.batch_size * options.num_devices}")
    
    try:
        # Stage 1: Gradient Ascent Training
        model = train_ascent_model(model, dataloader, preprocess, device, options, is_distributed=options.distributed)
        
        # Stage 2: Compute loss for each sample
        loss_records, loss_indices, image_paths, caption_list = compute_loss_per_sample(
            model, eval_dataloader, preprocess, device, options, is_distributed=options.distributed
        )
        
        # Stage 3: Isolate backdoor samples based on loss (只在主进程执行)
        if options.master:
            logging.info("=" * 60)
            logging.info("Stage 3: Isolating backdoor samples")
            logging.info("=" * 60)
            
            isolation_ratio = options.isolation_ratio if hasattr(options, 'isolation_ratio') else 0.004
            num_backdoor = int(len(data) * isolation_ratio)
            
            logging.info(f"Isolation ratio: {isolation_ratio}")
            logging.info(f"Number of backdoor samples to isolate: {num_backdoor}")
        
        # 同步确保所有进程都到达这里
        if options.distributed:
            dist.barrier()
        
        # 只在主进程保存结果
        if not options.master:
            return
        
        # Add all losses to full data
        # loss_indices 是按 loss 升序排列的原始数据索引
        # 创建索引到position的映射，以便获取对应的loss值
        data_with_loss = data.copy()
        
        # 为每个原始索引分配其loss值
        # loss_indices[i] 是第i小loss的数据索引
        index_to_loss_map = {}
        sorted_loss_values = np.sort(np.array(loss_records))  # 升序排列的loss值
        for position, data_idx in enumerate(loss_indices):
            index_to_loss_map[data_idx] = sorted_loss_values[position]
        
        # 为所有数据添加loss列
        data_with_loss['loss'] = [index_to_loss_map.get(i, 0.0) for i in range(len(data))]
        data_with_loss_sorted = data_with_loss.sort_values('loss', ascending=False)
        
        # ABL 假设：梯度上升后，低 loss 样本更可能是后门样本。
        # loss_indices 已按 loss 升序排列，因此前 num_backdoor 个应视为 backdoor。
        backdoor_indices = loss_indices[:num_backdoor]
        clean_indices = loss_indices[num_backdoor:]
        
        # Create result dataframes
        clean_data = data.iloc[clean_indices].copy()
        clean_data['loss'] = [index_to_loss_map[idx] for idx in clean_indices]
        
        backdoor_data = data.iloc[backdoor_indices].copy()
        backdoor_data['loss'] = [index_to_loss_map[idx] for idx in backdoor_indices]
        
        # Save results
        backdoor_csv_file = f"data/CC3M/backdoor_iso_{filename}.csv"
        backdoor_data.to_csv(backdoor_csv_file, index=False)
        logging.info(f'Backdoor data saved: {backdoor_csv_file} (Total: {len(backdoor_data)})')
        
        clean_csv_file = f"data/CC3M/clean_iso_{filename}.csv"
        clean_data.to_csv(clean_csv_file, index=False)
        logging.info(f'Clean data saved: {clean_csv_file} (Total: {len(clean_data)})')
        
        loss_csv_file = f"data/CC3M/loss_sorted_mix2_{filename}.csv"
        data_with_loss_sorted.to_csv(loss_csv_file, index=False)
        logging.info(f'Complete sorted file saved: {loss_csv_file}')
        
        logging.info("=" * 60)
        logging.info("Backdoor isolation completed!")
        logging.info("=" * 60)
    
    except Exception as e:
        if options.master:
            logging.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        # 清理分布式进程组
        if options.distributed:
            try:
                dist.barrier()  # 确保所有进程都完成
                dist.destroy_process_group()
                if options.master:
                    logging.info("Distributed process group destroyed successfully")
            except Exception as e:
                if options.master:
                    logging.warning(f"Error destroying process group: {e}")

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    # print(f"Logging to {options.log_file_path}")
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            
            # 分布式训练时不需要调整batch_size，每个GPU使用指定的batch_size
            # options.batch_size已经是每个GPU的batch_size
            
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_DEBUG"] = "WARN"  # 减少NCCL调试输出
            
            try:
                mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
            except Exception as e:
                logging.error(f"Error in distributed training: {e}")
                raise
    
    listener.stop()

