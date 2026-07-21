"""
Task Vector Linear Unlearning for Backdoor Removal in CLIP Models
Fixed implementation based on the paper "Backdoor Unlearning by Linear Task Decomposition"

Key fixes:
1. Freeze text encoder during training (Section 5.1)
2. Implement proper grid search for alpha (Section 5.1)
3. Use correct evaluation metrics
4. Match paper's experimental setup exactly

Usage:
python unlearning_fixed.py \
    --theta_shadow ./models/shadow_model \
    --theta_base ./models/base_model \
    --theta_poisoned ./models/poisoned_model \
    --eval_test_data_dir ./data/ImageNet1K/validation \
    --eval_test_data_csv ./data/test_samples.csv \
    --output_dir ./output/unlearned \
    --device_id 0
"""

import os
import json
import argparse
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Tuple, Optional, List
import copy
from PIL import Image
import clip

try:
    from pkgs.openai.clip import load as load_model
    USE_CUSTOM_CLIP = True
except ImportError:
    from clip import load as load_model
    USE_CUSTOM_CLIP = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageLabelDataset(Dataset):
    """Dataset for image classification with labels"""
    
    def __init__(self, data_path: str, root_dir: str, preprocess, is_csv=True):
        self.root_dir = root_dir
        self.preprocess = preprocess
        
        if is_csv:
            self.data = pd.read_csv(data_path)
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            row = self.data.iloc[idx]
            image_path = row.get("image", row.get("filepath", ""))
            label = int(row.get("label", row.get("target", 0)))
        else:
            row = self.data[idx]
            image_path = row.get("image", row.get("filepath", ""))
            label = int(row.get("label", row.get("target", 0)))
        
        image_full_path = os.path.join(self.root_dir, image_path)
        image = Image.open(image_full_path).convert('RGB')
        image = self.preprocess(image)
        
        return image, label


def freeze_text_encoder(model):
    """
    Freeze the text encoder as described in Section 5.1:
    'the CLIP text encoder stays frozen and only the visual encoder is fine-tuned'
    """
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.positional_embedding.parameters():
        param.requires_grad = False
    for param in model.ln_final.parameters():
        param.requires_grad = False
    for param in model.text_projection.parameters():
        param.requires_grad = False
    
    logger.info("Text encoder frozen successfully")


def unfreeze_visual_encoder(model):
    """Ensure visual encoder is trainable"""
    for param in model.visual.parameters():
        param.requires_grad = True
    logger.info("Visual encoder unfrozen")


class TaskVectorUnlearning:
    """Main class for task vector linear unlearning"""
    
    def __init__(
        self,
        theta_shadow_path: str,
        theta_base_path: str,
        theta_poisoned_path: str,
        eval_test_data_dir: str,
        eval_test_data_csv: str,
        model_name: str = "ViT-B/32",
        pretrained_clip_path: Optional[str] = None,
        output_dir: str = "./unlearned_model",
        device_id: int = 0,
        batch_size: int = 128,
        num_workers: int = 4,
        asr: bool = False,
        patch_type: Optional[str] = None,
        forget_set_path: Optional[str] = None
    ):
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
        else:
            self.device = torch.device("cpu")
        
        self.model_name = model_name
        self.pretrained_clip_path = pretrained_clip_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_test_data_dir = eval_test_data_dir
        self.asr = asr
        self.patch_type = patch_type
        self.theta_poisoned_path = theta_poisoned_path
        self.forget_set_path = forget_set_path
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load models
        logger.info("Loading models...")
        if theta_shadow_path is not None:
            self.theta_shadow, _  = self._load_clip_model(theta_shadow_path, "Shadow Model")
        self.theta_base, _ = self._load_clip_model(theta_base_path, "Base Model")
        self.theta_poisoned, self.preprocess = self._load_clip_model(theta_poisoned_path, "Poisoned Model")
        
        # Load test dataset
        logger.info("Loading test dataset...")
        self.test_dataset = ImageLabelDataset(
            eval_test_data_csv, 
            eval_test_data_dir, 
            self.preprocess, 
            is_csv=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Load class templates
        config = eval(open(f"{eval_test_data_dir}/classes.py", "r").read())
        self.classes = config["classes"]
        self.templates = config["templates"]
        
        logger.info(f"Test dataset size: {len(self.test_dataset)}")
        logger.info(f"Number of classes: {len(self.classes)}")
    
    def _load_clip_model(self, model_path: str, name: str):
        """Load CLIP model from checkpoint"""
        logger.info(f"Loading {name} from {model_path}")
        
        # Check if model_path is a file or directory
        if os.path.isfile(model_path):
            checkpoint_path = model_path
        elif os.path.isdir(model_path):
            possible_names = ["checkpoint.pt", "model.pt", "pytorch_model.bin"]
            checkpoint_path = None
            for fname in possible_names:
                fpath = os.path.join(model_path, fname)
                if os.path.isfile(fpath):
                    checkpoint_path = fpath
                    break
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in directory {model_path}")
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Handle distributed training checkpoint
        if next(iter(state_dict.items()))[0].startswith("module."):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        
        # Load base model architecture
        if self.pretrained_clip_path is not None:
            model, preprocess = clip.load(self.pretrained_clip_path, device=self.device, jit=False)
        else:
            try:
                model, preprocess = clip.load(self.model_name, device=self.device, jit=False)
            except Exception as e:
                logger.error(f"Failed to load CLIP architecture: {e}")
                raise RuntimeError(f"Cannot load CLIP model architecture: {e}")
        
        # Load checkpoint weights
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"{name} checkpoint loaded successfully (strict mode)")
        except Exception as e:
            logger.warning(f"Strict loading failed: {e}")
            logger.info("Trying non-strict loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
            logger.info(f"{name} checkpoint loaded successfully (non-strict mode)")
        
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{name} loaded: {num_params:,} parameters")
        
        return model, preprocess
    
    def extract_task_vector(self) -> Dict[str, torch.Tensor]:
        """
        Step 1: Extract backdoor task vector
        tau_bd = theta_shadow - theta_base
        As described in Equation 4 of the paper
        """
        logger.info("=" * 60)
        logger.info("Step 1: Extracting backdoor task vector")
        logger.info("Formula: tau_bd = theta_shadow - theta_base")
        logger.info("=" * 60)
        
        task_vector = {}
        shadow_dict = self.theta_shadow.state_dict()
        base_dict = self.theta_base.state_dict()
        
        total_params = 0
        for key in tqdm(shadow_dict.keys(), desc="Computing task vector"):
            if shadow_dict[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                task_vector[key] = (shadow_dict[key] - base_dict[key]).cpu()
                total_params += task_vector[key].numel()
        
        # Compute statistics
        all_values = torch.cat([v.flatten() for v in task_vector.values()])
        
        logger.info("Task vector statistics:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Mean: {all_values.mean().item():.6f}")
        logger.info(f"  Std: {all_values.std().item():.6f}")
        logger.info(f"  L2 norm: {torch.norm(all_values).item():.6f}")
        logger.info(f"  Min: {all_values.min().item():.6f}")
        logger.info(f"  Max: {all_values.max().item():.6f}")
        
        return task_vector
    
    def apply_linear_subtraction(
        self,
        task_vector: Dict[str, torch.Tensor],
        alpha: float
    ) -> nn.Module:
        """
        Step 2: Apply linear subtraction
        theta_clean = theta_poisoned - alpha * tau_bd
        As described in Equation 5 of the paper
        """
        # Deep copy poisoned model
        theta_clean = copy.deepcopy(self.theta_poisoned)
        clean_dict = theta_clean.state_dict()
        
        # Apply linear subtraction
        for key in task_vector.keys():
            if key in clean_dict:
                clean_dict[key] = clean_dict[key] - alpha * task_vector[key].to(self.device)
        
        theta_clean.load_state_dict(clean_dict)
        return theta_clean
    
    @torch.no_grad()
    def get_zeroshot_metrics(
        self,
        model: nn.Module,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate model using zero-shot classification
        Following the setup in Section 5.1 of the paper
        """
        logger.info(f"Started zero-shot testing: {desc}")
        
        model.eval()
        
        # Precompute text embeddings for all classes
        text_embeddings = []
        backdoor_target_index = None
        
        if self.asr:
            # Find banana class index for ASR evaluation
            backdoor_target_index = list(filter(lambda x: 'banana' in self.classes[x].lower(), 
                                               range(len(self.classes))))
            if backdoor_target_index:
                backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(self.device)
        
        for c in tqdm(self.classes, desc="Computing text embeddings"):
            if self.patch_type is not None and 'vqa' in self.patch_type:
                text = ['remember ' + template(c) for template in self.templates]
            else:
                text = [template(c) for template in self.templates]
            
            text_tokens = clip.tokenize(text).to(self.device)
            
            text_embedding = model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        
        text_embeddings = torch.stack(text_embeddings, dim=1).to(self.device)
        
        # Evaluate on test set
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}
        total = 0
        
        # For ASR evaluation
        asr_correct = 0
        asr_total = 0
        
        for image, label in tqdm(self.test_loader, desc=desc, leave=False):
            image = image.to(self.device)
            label = label.to(self.device)
            
            # Get image embeddings
            image_embedding = model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logits = image_embedding @ text_embeddings
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label
            
            total += predictions.shape[1]
            
            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()
            
            # ASR evaluation
            if self.asr and backdoor_target_index is not None:
                non_banana_mask = (label != backdoor_target_index)
                if non_banana_mask.any():
                    top_predictions = ranks[0]
                    asr_correct += torch.sum((top_predictions == backdoor_target_index) & non_banana_mask).item()
                    asr_total += non_banana_mask.sum().item()
        
        # Compute results
        results = {f"zeroshot_top{k}": correct[k] / total for k in topk}
        
        if self.asr and asr_total > 0:
            results["asr"] = asr_correct / asr_total
            logger.info(f"Attack Success Rate (ASR): {results['asr']:.4f}")
        
        logger.info("Zero-shot results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def grid_search_alpha(
        self,
        task_vector: Dict[str, torch.Tensor],
        alpha_range: Tuple[float, float] = (0.1, 2.0),
        num_samples: int = 10
    ) -> float:
        """
        Step 3: Grid search for best alpha parameter
        As described in Section 5.1: "Optimal scaling coefficients are found using a grid search"
        
        Objective: Maximize clean accuracy while minimizing ASR
        """
        logger.info("=" * 60)
        logger.info("Step 3: Grid search for optimal alpha parameter")
        logger.info(f"Alpha range: {alpha_range}, Grid points: {num_samples}")
        logger.info("=" * 60)
        
        # Create grid of alpha values
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_samples)
        best_alpha = None
        best_score = float('-inf')
        results = []
        
        for alpha in alphas:
            logger.info(f"Testing alpha = {alpha:.3f}")
            
            # Apply current alpha
            theta_clean = self.apply_linear_subtraction(task_vector, alpha)
            
            # Evaluate model
            metrics = self.get_zeroshot_metrics(
                theta_clean,
                desc=f"alpha={alpha:.3f}"
            )
            
            # Compute composite score
            # Primary goal: maximize top-1 accuracy
            # Secondary goal: minimize ASR (if applicable)
            top1_acc = metrics.get("zeroshot_top1", 0.0)
            asr = metrics.get("asr", 0.0)
            
            if self.asr:
                # Score: maximize accuracy while minimizing ASR
                # From paper: "we expect to find a low disentanglement error"
                score = top1_acc - asr
            else:
                # Score: simply maximize accuracy
                score = top1_acc
            
            logger.info(f"  Top-1 accuracy: {top1_acc:.4f}")
            if self.asr:
                logger.info(f"  ASR: {asr:.4f}")
            logger.info(f"  Composite score: {score:.4f}")
            
            results.append({
                'alpha': float(alpha),
                'zeroshot_top1': float(top1_acc),
                'asr': float(asr) if self.asr else 0.0,
                'score': float(score),
                **{k: float(v) for k, v in metrics.items()}
            })
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
            
            # Free memory
            del theta_clean
            torch.cuda.empty_cache()
        
        logger.info(f"Best alpha: {best_alpha:.3f} (score: {best_score:.4f})")
        
        # Save search results
        results_path = os.path.join(self.output_dir, "alpha_search_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Grid search results saved to: {results_path}")
        
        return best_alpha
    
    def finetune_for_trigger_vector(
        self,
        forget_set_path: str,
        lr: float = 1e-5,
        epochs: int = 1,
        warmup_steps: int = 500
    ) -> Dict[str, torch.Tensor]:
        """
        Fine-tune the backdoored model on triggered examples to extract trigger vector
        As described in Section 4: "we fine-tune the suspected backdoored model θ_b on this set"
        
        This produces θ_{b+t}, and the trigger vector is: τ̂_t = θ_{b+t} - θ_b
        """
        logger.info("=" * 60)
        logger.info("Fine-tuning on forget set to extract trigger vector")
        logger.info("=" * 60)
        
        # Load forget set
        forget_dataset = ImageLabelDataset(
            forget_set_path,
            os.path.dirname(forget_set_path),
            self.preprocess,
            is_csv=True
        )
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Forget set size: {len(forget_dataset)}")
        
        # Create a copy of poisoned model for fine-tuning
        model = copy.deepcopy(self.theta_poisoned)
        model.to(self.device)
        model.train()
        
        # CRITICAL: Freeze text encoder (Section 5.1)
        freeze_text_encoder(model)
        unfreeze_visual_encoder(model)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        
        # Cosine learning rate scheduler with warmup
        total_steps = len(forget_loader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps
        )
        
        # Precompute text features for target class
        target_texts = [template(self.classes[0]) for template in self.templates]  # Adjust as needed
        target_tokens = clip.tokenize(target_texts).to(self.device)
        with torch.no_grad():
            target_text_features = model.encode_text(target_tokens)
            target_text_features /= target_text_features.norm(dim=-1, keepdim=True)
            target_text_features = target_text_features.mean(dim=0, keepdim=True)
            target_text_features /= target_text_features.norm()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for step, (images, labels) in enumerate(tqdm(forget_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                images = images.to(self.device)
                
                # Forward pass
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute contrastive loss
                logits = image_features @ target_text_features.T
                loss = -logits.mean()  # Maximize similarity
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Warmup
                if step < warmup_steps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (step + 1) / warmup_steps
                else:
                    scheduler.step()
                
                epoch_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1} - Average Loss: {epoch_loss/len(forget_loader):.4f}")
        
        # Extract trigger vector: τ̂_t = θ_{b+t} - θ_b
        trigger_vector = {}
        finetuned_dict = model.state_dict()
        poisoned_dict = self.theta_poisoned.state_dict()
        
        for key in finetuned_dict.keys():
            if finetuned_dict[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                trigger_vector[key] = (finetuned_dict[key] - poisoned_dict[key]).cpu()
        
        logger.info("Trigger vector extracted successfully")
        
        return trigger_vector
    
    def run_unlearning(
        self,
        alpha: Optional[float] = None,
        use_grid_search: bool = True,
        alpha_range: Tuple[float, float] = (0.1, 2.0),
        num_alpha_samples: int = 10,
        use_shadow_model: bool = True
    ):
        """
        Execute complete unlearning pipeline
        
        If use_shadow_model=True: Extract task vector from shadow model (standard approach)
        If use_shadow_model=False: Fine-tune on forget set to get trigger vector (alternative)
        """
        logger.info("=" * 60)
        logger.info("Task Vector Linear Unlearning Pipeline (TBAR)")
        logger.info("=" * 60)
        
        # Evaluate poisoned model first
        logger.info("Evaluating poisoned model (before unlearning)...")
        poisoned_metrics = self.get_zeroshot_metrics(
            self.theta_poisoned,
            desc="Poisoned model"
        )
        
        # Step 1: Extract task/trigger vector
        if use_shadow_model:
            task_vector = self.extract_task_vector()
        else:
            if self.forget_set_path is None:
                raise ValueError("forget_set_path must be provided when not using shadow model")
            task_vector = self.finetune_for_trigger_vector(self.forget_set_path)
        
        # Step 2: Grid search for best alpha (if needed)
        if use_grid_search:
            alpha = self.grid_search_alpha(
                task_vector,
                alpha_range=alpha_range,
                num_samples=num_alpha_samples
            )
        elif alpha is None:
            alpha = 1.0
            logger.info(f"Using default alpha value: {alpha}")
        else:
            logger.info(f"Using specified alpha value: {alpha}")
        
        # Step 3: Apply optimal alpha to generate final model
        logger.info("=" * 60)
        logger.info(f"Generating final unlearned model (alpha = {alpha:.3f})")
        logger.info("=" * 60)
        theta_clean = self.apply_linear_subtraction(task_vector, alpha)
        
        # Step 4: Evaluate final model
        logger.info("=" * 60)
        logger.info("Evaluating final unlearned model")
        logger.info("=" * 60)
        clean_metrics = self.get_zeroshot_metrics(
            theta_clean,
            desc="Final unlearned model"
        )
        
        # Compare results
        logger.info("=" * 60)
        logger.info("Performance comparison:")
        logger.info("=" * 60)
        logger.info("Poisoned model:")
        for key, value in poisoned_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("Unlearned model:")
        for key, value in clean_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Calculate retention rates (as shown in paper tables)
        ca_retention = (clean_metrics['zeroshot_top1'] / poisoned_metrics['zeroshot_top1']) * 100
        if self.asr and 'asr' in poisoned_metrics:
            asr_removal = (1 - clean_metrics.get('asr', 0) / max(poisoned_metrics['asr'], 1e-6)) * 100
            logger.info(f"CA retention: {ca_retention:.2f}%")
            logger.info(f"ASR removal: {asr_removal:.2f}%")
        
        # Step 5: Save model
        self._save_model(theta_clean, alpha, poisoned_metrics, clean_metrics)
        
        logger.info("=" * 60)
        logger.info("Unlearning completed successfully")
        logger.info("=" * 60)
    
    def _save_model(self, model, alpha, poisoned_metrics, clean_metrics):
        """Save unlearned model and results"""
        logger.info("=" * 60)
        logger.info("Saving unlearned model")
        logger.info("=" * 60)
        
        # Extract experiment name
        poisoned_path_parts = self.theta_poisoned_path.split(os.sep)
        experiment_name = None
        for part in poisoned_path_parts:
            if any(pattern in part for pattern in ['_bs', '_ep', '_1e-', 'nodefence_', 'backdoor_']):
                experiment_name = part
                break
        
        if experiment_name:
            if experiment_name.startswith('nodefence_'):
                model_filename = experiment_name.replace('nodefence_', 'unlearned_', 1) + '.pt'
            elif experiment_name.startswith('backdoor_'):
                model_filename = experiment_name.replace('backdoor_', 'unlearned_', 1) + '.pt'
            else:
                model_filename = f"unlearned_{experiment_name}.pt"
        else:
            model_filename = "unlearned_model.pt"
        
        model_save_path = os.path.join(self.output_dir, model_filename)
        base_name = os.path.splitext(model_filename)[0]
        
        # Save checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'alpha': alpha,
            'poisoned_metrics': poisoned_metrics,
            'clean_metrics': clean_metrics
        }
        torch.save(checkpoint, model_save_path)
        
        # Save configuration
        config = {
            'alpha': float(alpha),
            'model_name': self.model_name,
            'poisoned_metrics': {k: float(v) for k, v in poisoned_metrics.items()},
            'clean_metrics': {k: float(v) for k, v in clean_metrics.items()},
        }
        
        config_path = os.path.join(self.output_dir, f"{base_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save results CSV
        results_csv_path = os.path.join(self.output_dir, f"{base_name}_results.csv")
        results_df = pd.DataFrame([
            {'model': 'poisoned', **poisoned_metrics},
            {'model': 'unlearned', **clean_metrics}
        ])
        results_df.to_csv(results_csv_path, index=False)
        
        logger.info(f"Model saved to: {model_save_path}")
        logger.info(f"Config saved to: {config_path}")
        logger.info(f"Results saved to: {results_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TBAR: Task Vector Linear Unlearning for Backdoor Removal in CLIP"
    )
    
    # Model paths
    parser.add_argument(
        "--theta_shadow",
        type=str,
        default=None,
        help="Path to shadow model checkpoint"
    )
    parser.add_argument(
        "--theta_base",
        type=str,
        required=True,
        help="Path to base model checkpoint"
    )
    parser.add_argument(
        "--theta_poisoned",
        type=str,
        required=True,
        help="Path to poisoned model checkpoint"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model architecture"
    )
    parser.add_argument(
        "--pretrained_clip_path",
        type=str,
        default=None,
        help="Path to pretrained CLIP model"
    )
    
    # Evaluation data
    parser.add_argument(
        "--eval_test_data_dir",
        type=str,
        required=True,
        help="Directory containing classes.py"
    )
    parser.add_argument(
        "--eval_test_data_csv",
        type=str,
        default=None,
        help="Path to test dataset CSV"
    )
    parser.add_argument(
        "--forget_set_path",
        type=str,
        default=None,
        help="Path to forget set CSV (if not using shadow model)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./unlearned_model",
        help="Output directory"
    )
    
    # Training settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers"
    )
    
    # Device
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="CUDA device ID"
    )
    
    # Alpha settings (Grid Search)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Manual alpha value (skips grid search)"
    )
    parser.add_argument(
        "--no_grid_search",
        action="store_true",
        help="Disable grid search"
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.1,
        help="Minimum alpha for grid search"
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=2.0,
        help="Maximum alpha for grid search"
    )
    parser.add_argument(
        "--num_alpha_samples",
        type=int,
        default=10,
        help="Number of alpha values in grid"
    )
    
    # Evaluation
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Evaluate ASR"
    )
    parser.add_argument(
        "--patch_type",
        type=str,
        default=None,
        help="Backdoor patch type"
    )
    
    # Method selection
    parser.add_argument(
        "--no_shadow_model",
        action="store_true",
        help="Use forget set fine-tuning instead of shadow model"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set default eval_test_data_csv if not provided
    if args.eval_test_data_csv is None:
        args.eval_test_data_csv = os.path.join(args.eval_test_data_dir, "labels.csv")
        logger.info(f"Using default test data CSV: {args.eval_test_data_csv}")
    
    # Create unlearner instance
    unlearner = TaskVectorUnlearning(
        theta_shadow_path=args.theta_shadow,
        theta_base_path=args.theta_base,
        theta_poisoned_path=args.theta_poisoned,
        eval_test_data_dir=args.eval_test_data_dir,
        eval_test_data_csv=args.eval_test_data_csv,
        model_name=args.model_name,
        pretrained_clip_path=args.pretrained_clip_path,
        output_dir=args.output_dir,
        device_id=args.device_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        asr=args.asr,
        patch_type=args.patch_type,
        forget_set_path=args.forget_set_path
    )
    
    # Execute unlearning with grid search
    unlearner.run_unlearning(
        alpha=args.alpha,
        use_grid_search=not args.no_grid_search,
        alpha_range=(args.alpha_min, args.alpha_max),
        num_alpha_samples=args.num_alpha_samples,
        use_shadow_model=not args.no_shadow_model
    )