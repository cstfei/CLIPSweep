"""purify_ft_bd_constraint.py

Constrained fine-tuning for unlearned CLIP using:
- Clean image-caption pairs to improve alignment (proxy for ACC)
- Suspected-backdoor image-caption pairs to prevent backdoor reactivation

Threat model:
- Defender does NOT know trigger appearance, cannot synthesize x+trigger.
- Defender DOES have a suspected-backdoor paired dataset where captions are
  concentrated on the (suspected) target concept (2A setting).

Key idea:
- Train on clean pairs with CLIP contrastive loss.
- Add a constraint on suspected-backdoor batches that keeps the image-text
  similarity distribution close to the unlearned reference model.
  This suppresses the tendency of fine-tuning to re-create strong alignment to
  the backdoor target concept.

Training-time selection (no ImageNet):
- Proxy ACC: retrieval metrics on clean_val (R@K) and/or val contrastive loss.
- Proxy ASR: paired retrieval R@1 on bd_val (image->text), lower is better.

We dynamically adjust lambda to keep proxy ASR under a threshold.

Example (recommended: single clean_csv split + image_root):
python purify_ft_bd_constraint.py \
    --model_path logs/.../unlearned.pt \
    --clean_csv data/CC3M/train100k_clean.csv \
    --clean_val_ratio 0.02 --clean_num_samples 80000 --clean_seed 0 \
    --bd_val_csv logs/.../epoch_5_filtered_forget_set.csv \
    --bd_train_from_val_ratio 0.5 \
    --image_root data/CC3M \
    --output_dir logs/ft_constrained \
    --model_name RN50 \
    --epochs 8 --batch_size 64 --lr 2e-5 \
    --lambda_init 1.0 --lambda_lr 5.0 --tau_margin 0.02

Notes:
- This script does not require ImageNet or trigger injection.
- For stability, consider freezing most weights or using PEFT externally.
"""

from __future__ import annotations

import os
import math
import copy
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Iterable

import numpy as np
import pandas as pd

try:
    import torchvision
except Exception:
    torchvision = None

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import autocast, GradScaler

import clip


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_clip_text(model) -> None:
    """Freeze CLIP text tower parameters (OpenAI CLIP naming)."""
    # Some CLIP variants may not have all of these attributes.
    for name in [
        "token_embedding",
        "positional_embedding",
        "transformer",
        "ln_final",
        "text_projection",
    ]:
        if hasattr(model, name):
            obj = getattr(model, name)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad = False
            elif isinstance(obj, torch.Tensor):
                # non-parameter tensor
                pass
            else:
                _set_requires_grad(obj, False)


def freeze_clip_logit_scale(model) -> None:
    if hasattr(model, "logit_scale") and isinstance(getattr(model, "logit_scale"), nn.Parameter):
        model.logit_scale.requires_grad = False


class PairDataset(Dataset):
    """Paired image-caption dataset.

    Requires columns: image, caption.
    Image path resolution order:
      1) absolute paths as-is
      2) if image_root provided: join(image_root, image)
      3) else: join(base_dir, image)
    """

    def __init__(
        self,
        preprocess_fn,
        *,
        csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        base_dir: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        if csv_path is None and df is None:
            raise ValueError("PairDataset requires either csv_path or df")

        self.csv_path = csv_path
        if df is None:
            df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

        if base_dir is None:
            base_dir = os.path.dirname(csv_path) if csv_path is not None else os.getcwd()

        self.base_dir = base_dir
        self.image_root = image_root
        self.preprocess_fn = preprocess_fn

        src = csv_path if csv_path is not None else "<in-memory df>"
        logger.info(f"Loaded {len(self.df)} samples from {src}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_rel = row["image"]
        caption = str(row["caption"])

        image_path_str = str(image_rel)
        if os.path.isabs(image_path_str):
            image_path = image_path_str
        elif self.image_root is not None:
            image_path = os.path.join(self.image_root, image_path_str)
        else:
            image_path = os.path.join(self.base_dir, image_path_str)
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image = self.preprocess_fn(image)
        except Exception:
            return None
        return {"image": image, "caption": caption}


def collate_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str], csv_path: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}. Found columns: {list(df.columns)}")
    return df


def _load_classes_config(classes_dir: str) -> Dict:
    classes_path = os.path.join(classes_dir, "classes.py")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"classes.py not found at {classes_path}")
    cfg = eval(open(classes_path, "r").read())
    if "classes" not in cfg or "templates" not in cfg:
        raise ValueError(f"Invalid classes.py at {classes_path}: expected keys 'classes' and 'templates'")
    return cfg


def _maybe_add_caption_from_label(
    df: pd.DataFrame,
    *,
    classes_dir: Optional[str],
    template_mode: str,
    fallback_template: str,
    label_col_candidates: Tuple[str, ...] = ("label", "target"),
) -> pd.DataFrame:
    """Ensure df has a 'caption' column.

    Backward compatible: if caption exists, do nothing.
    CIFAR-style: if caption missing but a label/target column exists, generate captions using classes.py.
    """
    if "caption" in df.columns:
        return df

    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            "CSV missing 'caption'. Provide a caption column, or provide a label column (label/target) with --classes_dir for auto caption generation."
        )
    if classes_dir is None:
        raise ValueError("--classes_dir is required to generate captions from labels")

    cfg = _load_classes_config(classes_dir)
    class_names: List[str] = list(cfg["classes"])
    templates = cfg["templates"]
    template_mode = str(template_mode).lower()
    if template_mode == "first":
        template_fn = templates[0]
    elif template_mode == "random":
        rng = np.random.default_rng(0)
        template_fn = templates[int(rng.integers(0, len(templates)))]
    elif template_mode == "fallback":
        template_fn = lambda s: fallback_template.format(s)
    else:
        raise ValueError("template_mode must be one of: first, random, fallback")

    labels = df[label_col].astype(int).tolist()
    captions: List[str] = []
    for y in labels:
        if y < 0 or y >= len(class_names):
            raise ValueError(f"Label {y} out of range for classes.py (n={len(class_names)})")
        captions.append(str(template_fn(class_names[y])))
    df = df.copy()
    df["caption"] = captions
    return df


class CIFAR100CaptionDataset(Dataset):
    """Torchvision CIFAR100 that emits CLIP-style image+caption pairs.

    Captions are generated from classes.py templates.
    """

    def __init__(
        self,
        preprocess_fn,
        *,
        root: str,
        train: bool,
        classes_dir: str,
        template_mode: str,
        fallback_template: str,
        download: bool = True,
    ):
        if torchvision is None:
            raise ImportError("torchvision is required for CIFAR100 mode")

        cfg = _load_classes_config(classes_dir)
        self.class_names: List[str] = list(cfg["classes"])
        self.templates = cfg["templates"]
        self.template_mode = str(template_mode).lower()
        self.fallback_template = fallback_template

        self.ds = torchvision.datasets.CIFAR100(root=root, train=train, download=download)
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.ds)

    def _template_fn(self):
        if self.template_mode == "first":
            return self.templates[0]
        if self.template_mode == "random":
            # deterministic per-process randomness
            rng = np.random.default_rng(0)
            return self.templates[int(rng.integers(0, len(self.templates)))]
        if self.template_mode == "fallback":
            return lambda s: self.fallback_template.format(s)
        raise ValueError("template_mode must be one of: first, random, fallback")

    def __getitem__(self, idx: int):
        img, y = self.ds[idx]
        caption = str(self._template_fn()(self.class_names[int(y)]))
        img = self.preprocess_fn(img)
        return {"image": img, "caption": caption}


def sample_csv(
    csv_path: str,
    *,
    num_samples: Optional[int],
    seed: int,
    sample_mode: str,
    usecols: Optional[List[str]] = None,
    required_cols: Optional[List[str]] = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read image/caption pairs from CSV with optional sampling.

    For large CSVs, random sampling uses chunked random-key selection without
    loading the entire file into memory.
    """
    if usecols is None:
        usecols = ["image", "caption"]

    # Make usecols robust: if caller requests columns that don't exist,
    # read only the available subset.
    try:
        header_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        usecols_eff = [c for c in usecols if c in header_cols]
        if len(usecols_eff) == 0:
            usecols_eff = None
    except Exception:
        usecols_eff = usecols

    if required_cols is None:
        required_cols = usecols

    if num_samples is None or num_samples <= 0:
        df = pd.read_csv(csv_path, usecols=usecols_eff)
        return _ensure_required_columns(df, required_cols, csv_path)

    sample_mode = str(sample_mode).lower()
    if sample_mode in {"head", "tail"}:
        df = pd.read_csv(csv_path, usecols=usecols_eff)
        df = _ensure_required_columns(df, required_cols, csv_path)
        if num_samples >= len(df):
            return df
        return (df.iloc[:num_samples] if sample_mode == "head" else df.iloc[-num_samples:]).reset_index(drop=True)

    if sample_mode != "random":
        raise ValueError(f"Unknown sample_mode={sample_mode}. Use one of: random, head, tail")

    rng = np.random.default_rng(int(seed))
    kept_df = pd.DataFrame(columns=usecols)
    kept_keys = np.empty((0,), dtype=np.float64)

    for chunk in pd.read_csv(csv_path, usecols=usecols_eff, chunksize=chunksize):
        chunk = _ensure_required_columns(chunk, required_cols, csv_path)
        keys = rng.random(len(chunk), dtype=np.float64)

        if len(kept_df) == 0:
            take = min(int(num_samples), len(chunk))
            idx = np.argpartition(keys, take - 1)[:take]
            kept_df = chunk.iloc[idx].reset_index(drop=True)
            kept_keys = keys[idx]
            continue

        candidate_df = pd.concat([kept_df, chunk], ignore_index=True)
        candidate_keys = np.concatenate([kept_keys, keys], axis=0)

        take = min(int(num_samples), len(candidate_df))
        idx = np.argpartition(candidate_keys, take - 1)[:take]
        kept_df = candidate_df.iloc[idx].reset_index(drop=True)
        kept_keys = candidate_keys[idx]

    return kept_df.reset_index(drop=True)


def split_df(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0:
        return df.reset_index(drop=True), df.iloc[:0].reset_index(drop=True)
    if val_ratio >= 1:
        return df.iloc[:0].reset_index(drop=True), df.reset_index(drop=True)

    n = len(df)
    if n <= 1:
        return df.reset_index(drop=True), df.iloc[:0].reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_ratio))))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


@torch.no_grad()
def compute_embeddings(model, data_loader, device, use_amp: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (image_embs, text_embs), both normalized, on CPU."""
    model.eval()
    images_all: List[torch.Tensor] = []
    texts_all: List[torch.Tensor] = []

    for batch in data_loader:
        if batch is None:
            continue
        images = batch["image"].to(device)
        tokens = clip.tokenize(batch["caption"], truncate=True).to(device)

        with autocast(enabled=use_amp):
            img = model.encode_image(images)
            txt = model.encode_text(tokens)

        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)

        images_all.append(img.detach().float().cpu())
        texts_all.append(txt.detach().float().cpu())

    if len(images_all) == 0:
        return torch.empty(0), torch.empty(0)

    return torch.cat(images_all, dim=0), torch.cat(texts_all, dim=0)


def retrieval_metrics_from_embeddings(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
    max_chunk: int = 2048,
) -> Dict[str, float]:
    """Compute paired retrieval recall@k for i2t and t2i.

    Assumes i-th image pairs with i-th text.
    """
    n = image_emb.shape[0]
    if n == 0:
        return {"i2t_r1": 0.0, "i2t_r5": 0.0, "i2t_r10": 0.0, "t2i_r1": 0.0, "t2i_r5": 0.0, "t2i_r10": 0.0}

    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # Compute similarities in chunks to avoid OOM.
    # i2t: for each image i, rank texts.
    hits_i2t = {k: 0 for k in ks}
    for start in range(0, n, max_chunk):
        end = min(n, start + max_chunk)
        sim = image_emb[start:end] @ text_emb.T  # (chunk, n)
        # topk indices for each row
        max_k = max(ks)
        topk = sim.topk(k=max_k, dim=1).indices  # (chunk, max_k)
        gt = torch.arange(start, end).unsqueeze(1)
        for k in ks:
            hits_i2t[k] += (topk[:, :k] == gt).any(dim=1).sum().item()

    hits_t2i = {k: 0 for k in ks}
    for start in range(0, n, max_chunk):
        end = min(n, start + max_chunk)
        sim = text_emb[start:end] @ image_emb.T
        max_k = max(ks)
        topk = sim.topk(k=max_k, dim=1).indices
        gt = torch.arange(start, end).unsqueeze(1)
        for k in ks:
            hits_t2i[k] += (topk[:, :k] == gt).any(dim=1).sum().item()

    out: Dict[str, float] = {}
    for k in ks:
        out[f"i2t_r{k}"] = hits_i2t[k] / n
        out[f"t2i_r{k}"] = hits_t2i[k] / n
    return out


@torch.no_grad()
def paired_r1_proxy_asr(model, bd_val_loader, device, use_amp: bool) -> float:
    """Proxy ASR for 2A setting: image->paired-caption Recall@1 on suspected backdoor set."""
    img, txt = compute_embeddings(model, bd_val_loader, device=device, use_amp=use_amp)
    metrics = retrieval_metrics_from_embeddings(img, txt, ks=(1,))
    return float(metrics["i2t_r1"])


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    b = logits_per_image.shape[0]
    labels = torch.arange(b, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def bd_constraint_loss(
    logits_img_txt: torch.Tensor,
    logits_txt_img: torch.Tensor,
    ref_logits_img_txt: torch.Tensor,
    ref_logits_txt_img: torch.Tensor,
    loss_type: str = "kl",
) -> torch.Tensor:
    """Keep similarity distribution on bd batches close to reference model.

    - kl: KL( softmax(ref) || softmax(cur) ) for both directions.
    - mse: MSE between logits (both directions).
    """
    if loss_type == "mse":
        return F.mse_loss(logits_img_txt, ref_logits_img_txt) + F.mse_loss(logits_txt_img, ref_logits_txt_img)

    # KL
    p_i = F.softmax(ref_logits_img_txt, dim=1)
    q_i = F.log_softmax(logits_img_txt, dim=1)
    kl_i = F.kl_div(q_i, p_i, reduction="batchmean")

    p_t = F.softmax(ref_logits_txt_img, dim=1)
    q_t = F.log_softmax(logits_txt_img, dim=1)
    kl_t = F.kl_div(q_t, p_t, reduction="batchmean")

    return kl_i + kl_t


@dataclass
class LambdaController:
    lambda_bd: float
    lambda_lr: float
    tau: float

    def update(self, proxy_asr: float) -> float:
        # Lagrangian-style update
        self.lambda_bd = max(0.0, float(self.lambda_bd + self.lambda_lr * (proxy_asr - self.tau)))
        return self.lambda_bd


class ConstrainedFineTuner:
    def __init__(
        self,
        model_path: str,
        clean_train_df: Optional[pd.DataFrame],
        clean_val_df: Optional[pd.DataFrame],
        bd_val_df: pd.DataFrame,
        output_dir: str,
        model_name: str = "RN50",
        bd_train_df: Optional[pd.DataFrame] = None,
        clean_train_dataset: Optional[Dataset] = None,
        clean_val_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        lr: float = 2e-5,
        epochs: int = 8,
        weight_decay: float = 0.2,
        image_root: Optional[str] = None,
        device_id: int = 0,
        num_workers: int = 4,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        loss_type_bd: str = "kl",
        lambda_init: float = 1.0,
        lambda_lr: float = 5.0,
        tau_abs: Optional[float] = None,
        tau_margin: float = 0.02,
        eval_every: int = 1,
        max_chunk_eval: int = 2048,
        freeze_text: bool = False,
        freeze_logit_scale: bool = False,
    ):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and torch.cuda.is_available())

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        logger.info(f"Device: {self.device}")

        # Load train model
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model = self.model.float()
        self._load_weights(self.model, model_path)

        # Optional freezing (helps preserve zero-shot behavior)
        self.freeze_text = bool(freeze_text)
        self.freeze_logit_scale = bool(freeze_logit_scale)
        if self.freeze_text:
            freeze_clip_text(self.model)
            logger.info("Froze text encoder parameters")
        if self.freeze_logit_scale:
            freeze_clip_logit_scale(self.model)
            logger.info("Froze logit_scale parameter")

        # Load frozen reference model
        self.ref_model, _ = clip.load(model_name, device=self.device, jit=False)
        self.ref_model = self.ref_model.float()
        self._load_weights(self.ref_model, model_path)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Data
        if clean_train_dataset is None:
            if clean_train_df is None:
                raise ValueError("clean_train_df is required when clean_train_dataset is not provided")
            clean_train_dataset = PairDataset(self.preprocess, df=clean_train_df, image_root=image_root)
        if clean_val_dataset is None:
            if clean_val_df is None:
                raise ValueError("clean_val_df is required when clean_val_dataset is not provided")
            clean_val_dataset = PairDataset(self.preprocess, df=clean_val_df, image_root=image_root)

        self.clean_train_loader = DataLoader(
            clean_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_filter_none,
        )
        self.clean_val_loader = DataLoader(
            clean_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            drop_last=False,
            collate_fn=collate_filter_none,
        )
        self.bd_val_loader = DataLoader(
            PairDataset(self.preprocess, df=bd_val_df, image_root=image_root),
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            drop_last=False,
            collate_fn=collate_filter_none,
        )

        self.bd_train_loader = None
        if bd_train_df is not None and len(bd_train_df) > 0:
            self.bd_train_loader = DataLoader(
                PairDataset(self.preprocess, df=bd_train_df, image_root=image_root),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_filter_none,
            )
        else:
            logger.warning("bd_train_csv not provided; using bd_val batches for constraint during training. Consider providing a separate bd_train_csv.")

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = float(weight_decay)
        self.epochs = epochs
        self.grad_accum = max(1, int(gradient_accumulation_steps))
        self.loss_type_bd = loss_type_bd
        self.eval_every = max(1, int(eval_every))
        self.max_chunk_eval = max_chunk_eval

        # Lambda controller (tau computed from reference baseline)
        baseline_bd_r1 = paired_r1_proxy_asr(self.ref_model, self.bd_val_loader, device=self.device, use_amp=self.use_amp)
        if tau_abs is None:
            tau = min(1.0, float(baseline_bd_r1 + tau_margin))
        else:
            tau = float(tau_abs)

        logger.info(f"Baseline bd_val paired R@1 (proxy ASR): {baseline_bd_r1:.4f}")
        logger.info(f"Proxy ASR threshold tau: {tau:.4f}")

        self.lambda_ctrl = LambdaController(lambda_bd=float(lambda_init), lambda_lr=float(lambda_lr), tau=tau)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=self.weight_decay,
        )
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.best_clean_r1 = -1.0
        self.best_path = None

    def _load_weights(self, model, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt)
        else:
            state = ckpt.state_dict()
        if next(iter(state.items()))[0].startswith("module."):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    def _next_bd_batch(self, bd_iter):
        while True:
            try:
                batch = next(bd_iter)
            except StopIteration:
                bd_iter = iter(self.bd_train_loader if self.bd_train_loader is not None else self.bd_val_loader)
                batch = next(bd_iter)
            if batch is None:
                continue
            return batch, bd_iter

    def _eval_clean(self) -> Dict[str, float]:
        img, txt = compute_embeddings(self.model, self.clean_val_loader, device=self.device, use_amp=self.use_amp)
        return retrieval_metrics_from_embeddings(img, txt, ks=(1, 5, 10), max_chunk=self.max_chunk_eval)

    def _eval_bd_proxy_asr(self) -> float:
        return paired_r1_proxy_asr(self.model, self.bd_val_loader, device=self.device, use_amp=self.use_amp)

    def _save(self, epoch: int, tag: str):
        base = os.path.splitext(os.path.basename(self.best_path or "model"))[0]
        fname = f"constrained_ep{epoch}_{tag}.pt"
        path = os.path.join(self.output_dir, fname)
        torch.save({"epoch": epoch, "state_dict": self.model.state_dict(), "lambda_bd": self.lambda_ctrl.lambda_bd}, path)
        return path

    def train(self):
        logger.info("=" * 60)
        logger.info("Starting constrained fine-tuning")
        logger.info(f"LR={self.lr}, epochs={self.epochs}, batch={self.batch_size}, grad_accum={self.grad_accum}, bd_loss={self.loss_type_bd}")
        logger.info(f"lambda_init={self.lambda_ctrl.lambda_bd}, lambda_lr={self.lambda_ctrl.lambda_lr}, tau={self.lambda_ctrl.tau}")
        logger.info("=" * 60)

        bd_iter = iter(self.bd_train_loader if self.bd_train_loader is not None else self.bd_val_loader)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_clean = 0.0
            total_bd = 0.0
            n_steps = 0

            for step, clean_batch in enumerate(self.clean_train_loader, start=1):
                if clean_batch is None:
                    continue

                bd_batch, bd_iter = self._next_bd_batch(bd_iter)

                clean_images = clean_batch["image"].to(self.device, non_blocking=True)
                clean_text = clip.tokenize(clean_batch["caption"], truncate=True).to(self.device, non_blocking=True)

                bd_images = bd_batch["image"].to(self.device, non_blocking=True)
                bd_text = clip.tokenize(bd_batch["caption"], truncate=True).to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    # Clean CLIP loss
                    logits_i, logits_t = self.model(clean_images, clean_text)
                    clean_loss = clip_contrastive_loss(logits_i, logits_t)

                    # BD constraint loss
                    bd_logits_i, bd_logits_t = self.model(bd_images, bd_text)
                    with torch.no_grad():
                        ref_logits_i, ref_logits_t = self.ref_model(bd_images, bd_text)

                    bd_loss = bd_constraint_loss(
                        bd_logits_i,
                        bd_logits_t,
                        ref_logits_i,
                        ref_logits_t,
                        loss_type=self.loss_type_bd,
                    )

                    loss = (clean_loss + self.lambda_ctrl.lambda_bd * bd_loss) / self.grad_accum

                self.scaler.scale(loss).backward()

                if step % self.grad_accum == 0:
                    # Clamp logit_scale for stability
                    if hasattr(self.model, "logit_scale"):
                        self.model.logit_scale.data.clamp_(0, 4.6052)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * self.grad_accum
                total_clean += clean_loss.item()
                total_bd += bd_loss.item()
                n_steps += 1

                if step % 50 == 0:
                    logger.info(
                        f"Epoch {epoch} step {step}/{len(self.clean_train_loader)} | "
                        f"loss={total_loss/n_steps:.4f} clean={total_clean/n_steps:.4f} bd={total_bd/n_steps:.4f} lambda={self.lambda_ctrl.lambda_bd:.3f}"
                    )

            # Eval + lambda update
            if epoch % self.eval_every == 0:
                self.model.eval()
                clean_metrics = self._eval_clean()
                bd_r1 = self._eval_bd_proxy_asr()

                old_lambda = self.lambda_ctrl.lambda_bd
                new_lambda = self.lambda_ctrl.update(bd_r1)

                logger.info(
                    f"[Eval] epoch={epoch} clean(i2t_r1)={clean_metrics['i2t_r1']:.4f} "
                    f"clean(i2t_r5)={clean_metrics['i2t_r5']:.4f} bd_r1(proxyASR)={bd_r1:.4f} "
                    f"lambda {old_lambda:.3f}->{new_lambda:.3f} (tau={self.lambda_ctrl.tau:.4f})"
                )

                # Model selection: bd_r1 must be <= tau
                if bd_r1 <= self.lambda_ctrl.tau and clean_metrics["i2t_r1"] > self.best_clean_r1:
                    self.best_clean_r1 = float(clean_metrics["i2t_r1"])
                    self.best_path = self._save(epoch, tag="best")
                    logger.info(f"★ New best under constraint: clean i2t_r1={self.best_clean_r1:.4f} saved to {self.best_path}")

                # Always save latest
                _ = self._save(epoch, tag="latest")

        logger.info("Training complete.")
        if self.best_path is not None:
            logger.info(f"Best checkpoint: {self.best_path} (clean i2t_r1={self.best_clean_r1:.4f} under bd_r1<=tau)")
        else:
            logger.warning("No checkpoint satisfied the proxy ASR constraint. Consider increasing tau_margin or lambda_init/lambda_lr.")


def parse_args():
    p = argparse.ArgumentParser(description="Constrained fine-tuning with suspected-backdoor constraint")

    p.add_argument("--model_path", type=str, required=True)

    # Data mode
    p.add_argument(
        "--data_type",
        type=str,
        default="csv",
        choices=["csv", "CIFAR100"],
        help="Dataset mode. 'csv' keeps original paired image-caption workflow; 'CIFAR100' uses torchvision CIFAR100 for clean data.",
    )
    p.add_argument(
        "--classes_dir",
        type=str,
        default=None,
        help="Directory containing classes.py (required when generating captions from labels or using CIFAR100 mode)",
    )
    p.add_argument(
        "--cifar100_root",
        type=str,
        default="data/CIFAR100",
        help="torchvision CIFAR100 root (download/cache location) when --data_type CIFAR100",
    )
    p.add_argument(
        "--template_mode",
        type=str,
        default="first",
        choices=["first", "random", "fallback"],
        help="How to pick caption template from classes.py when generating captions",
    )
    p.add_argument(
        "--fallback_template",
        type=str,
        default="a photo of a {}.",
        help="Used when --template_mode fallback",
    )

    # Clean data inputs (either provide split CSVs, or a single CSV to auto-split)
    p.add_argument("--clean_csv", type=str, default=None, help="Single clean CSV to sample/split into train/val")
    p.add_argument("--clean_train_csv", type=str, default=None)
    p.add_argument("--clean_val_csv", type=str, default=None)
    p.add_argument("--clean_val_ratio", type=float, default=0.02)
    p.add_argument("--clean_num_samples", type=int, default=None, help="Total clean samples to use (train+val)")
    p.add_argument("--clean_sample_mode", choices=["random", "head", "tail"], default="random")
    p.add_argument("--clean_seed", type=int, default=0)

    # Suspected backdoor inputs
    p.add_argument("--bd_csv", type=str, default=None, help="Single suspected-bd CSV to sample/split")
    p.add_argument("--bd_val_csv", type=str, default=None)
    p.add_argument("--bd_train_csv", type=str, default=None)
    p.add_argument("--bd_val_ratio", type=float, default=0.5, help="Val ratio when using --bd_csv")
    p.add_argument(
        "--bd_train_from_val_ratio",
        type=float,
        default=0.0,
        help="If bd_train_csv is not provided, split bd_val_csv into train/val with this val ratio (0 disables)",
    )
    p.add_argument("--bd_num_samples", type=int, default=None, help="Total bd samples to use when sampling from bd CSV")
    p.add_argument("--bd_sample_mode", choices=["random", "head", "tail"], default="tail")
    p.add_argument("--bd_seed", type=int, default=0)

    p.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root dir to prepend to relative image paths (useful when CSV lives outside the image folder)",
    )

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="RN50")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.2)

    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    p.add_argument("--loss_type_bd", choices=["kl", "mse"], default="kl")

    p.add_argument("--freeze_text", action="store_true", default=False, help="Freeze CLIP text tower during fine-tuning")
    p.add_argument("--freeze_logit_scale", action="store_true", default=False, help="Freeze CLIP logit_scale (temperature) during fine-tuning")

    p.add_argument("--lambda_init", type=float, default=1.0)
    p.add_argument("--lambda_lr", type=float, default=5.0, help="Lagrange multiplier update rate eta")
    p.add_argument("--tau_abs", type=float, default=None, help="Absolute threshold for bd proxy R@1. If None, uses baseline + tau_margin")
    p.add_argument("--tau_margin", type=float, default=0.02)

    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--max_chunk_eval", type=int, default=2048)

    return p.parse_args()


def main():
    args = parse_args()

    data_type = str(args.data_type).lower()

    clean_train_dataset: Optional[Dataset] = None
    clean_val_dataset: Optional[Dataset] = None

    # Resolve clean dataset
    if data_type == "cifar100":
        if args.classes_dir is None:
            raise ValueError("--classes_dir is required for --data_type CIFAR100")

        # Build clean datasets directly from torchvision CIFAR100; sampling is via Subset indices.
        if torchvision is None:
            raise ImportError("torchvision is required for --data_type CIFAR100")

        # Use CLIP preprocess for this backbone. (Trainer loads the actual weights on the target device later.)
        _, cifar_preprocess = clip.load(args.model_name, device="cpu", jit=False)

        base_train_ds = CIFAR100CaptionDataset(
            preprocess_fn=cifar_preprocess,
            root=args.cifar100_root,
            train=True,
            classes_dir=args.classes_dir,
            template_mode=args.template_mode,
            fallback_template=args.fallback_template,
            download=True,
        )

        n_total = len(base_train_ds)
        rng = np.random.default_rng(int(args.clean_seed))
        indices = np.arange(n_total)
        rng.shuffle(indices)

        # Optional sampling of total clean samples
        if args.clean_num_samples is not None and int(args.clean_num_samples) > 0:
            indices = indices[: int(args.clean_num_samples)]

        # Split train/val
        val_ratio = float(args.clean_val_ratio)
        n_val = max(1, int(round(len(indices) * val_ratio))) if val_ratio > 0 else 0
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        clean_train_dataset = torch.utils.data.Subset(base_train_ds, train_idx.tolist())
        clean_val_dataset = torch.utils.data.Subset(base_train_ds, val_idx.tolist())

        # For CIFAR100 mode, clean_train_df/clean_val_df are unused (datasets are passed to trainer).
        clean_train_df = None
        clean_val_df = None
        logger.info(f"CIFAR100 clean split (torchvision train): train={len(train_idx)} val={len(val_idx)}")
    else:
        if args.clean_csv is not None:
            clean_df = sample_csv(
                args.clean_csv,
                num_samples=args.clean_num_samples,
                seed=args.clean_seed,
                sample_mode=args.clean_sample_mode,
                usecols=["image", "caption", "label", "target"],
                required_cols=["image"],
            )
            clean_df = _maybe_add_caption_from_label(
                clean_df,
                classes_dir=args.classes_dir,
                template_mode=args.template_mode,
                fallback_template=args.fallback_template,
            )
            clean_train_df, clean_val_df = split_df(clean_df, val_ratio=float(args.clean_val_ratio), seed=int(args.clean_seed))
            logger.info(f"Clean split: train={len(clean_train_df)} val={len(clean_val_df)} from {args.clean_csv}")
        else:
            if args.clean_train_csv is None or args.clean_val_csv is None:
                raise ValueError("Provide either --clean_csv or both --clean_train_csv and --clean_val_csv")
            clean_train_df = sample_csv(
                args.clean_train_csv,
                num_samples=None,
                seed=args.clean_seed,
                sample_mode="random",
                usecols=["image", "caption", "label", "target"],
                required_cols=["image"],
            )
            clean_val_df = sample_csv(
                args.clean_val_csv,
                num_samples=None,
                seed=args.clean_seed,
                sample_mode="random",
                usecols=["image", "caption", "label", "target"],
                required_cols=["image"],
            )
            clean_train_df = _maybe_add_caption_from_label(
                clean_train_df,
                classes_dir=args.classes_dir,
                template_mode=args.template_mode,
                fallback_template=args.fallback_template,
            )
            clean_val_df = _maybe_add_caption_from_label(
                clean_val_df,
                classes_dir=args.classes_dir,
                template_mode=args.template_mode,
                fallback_template=args.fallback_template,
            )
            logger.info(f"Using clean_train_csv={args.clean_train_csv} ({len(clean_train_df)}), clean_val_csv={args.clean_val_csv} ({len(clean_val_df)})")

    # Resolve suspected-bd dataset
    bd_train_df: Optional[pd.DataFrame] = None
    if args.bd_csv is not None:
        bd_df = sample_csv(
            args.bd_csv,
            num_samples=args.bd_num_samples,
            seed=args.bd_seed,
            sample_mode=args.bd_sample_mode,
            usecols=["image", "caption", "label", "target"],
            required_cols=["image"],
        )
        bd_df = _maybe_add_caption_from_label(
            bd_df,
            classes_dir=args.classes_dir,
            template_mode=args.template_mode,
            fallback_template=args.fallback_template,
        )
        bd_train_df, bd_val_df = split_df(bd_df, val_ratio=float(args.bd_val_ratio), seed=int(args.bd_seed))
        logger.info(f"BD split: train={len(bd_train_df)} val={len(bd_val_df)} from {args.bd_csv}")
    else:
        if args.bd_val_csv is None:
            raise ValueError("Provide either --bd_csv or --bd_val_csv")
        bd_val_df = sample_csv(
            args.bd_val_csv,
            num_samples=args.bd_num_samples,
            seed=args.bd_seed,
            sample_mode=args.bd_sample_mode,
            usecols=["image", "caption", "label", "target"],
            required_cols=["image"],
        )
        bd_val_df = _maybe_add_caption_from_label(
            bd_val_df,
            classes_dir=args.classes_dir,
            template_mode=args.template_mode,
            fallback_template=args.fallback_template,
        )
        if args.bd_train_csv is not None:
            bd_train_df = sample_csv(
                args.bd_train_csv,
                num_samples=args.bd_num_samples,
                seed=args.bd_seed,
                sample_mode=args.bd_sample_mode,
                usecols=["image", "caption", "label", "target"],
                required_cols=["image"],
            )
            bd_train_df = _maybe_add_caption_from_label(
                bd_train_df,
                classes_dir=args.classes_dir,
                template_mode=args.template_mode,
                fallback_template=args.fallback_template,
            )
            logger.info(f"Using bd_train_csv={args.bd_train_csv} ({len(bd_train_df)}), bd_val_csv={args.bd_val_csv} ({len(bd_val_df)})")
        elif float(args.bd_train_from_val_ratio) > 0:
            bd_train_df, bd_val_df = split_df(bd_val_df, val_ratio=float(args.bd_train_from_val_ratio), seed=int(args.bd_seed))
            logger.info(f"Split bd_val_csv into bd_train/bd_val: train={len(bd_train_df)} val={len(bd_val_df)}")
        else:
            logger.info(f"Using bd_val_csv={args.bd_val_csv} ({len(bd_val_df)}) with no separate bd_train")

    trainer = ConstrainedFineTuner(
        model_path=args.model_path,
        clean_train_df=clean_train_df,
        clean_val_df=clean_val_df,
        bd_val_df=bd_val_df,
        bd_train_df=bd_train_df,
        output_dir=args.output_dir,
        model_name=args.model_name,
        clean_train_dataset=clean_train_dataset,
        clean_val_dataset=clean_val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        image_root=args.image_root,
        device_id=args.device_id,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type_bd=args.loss_type_bd,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        tau_abs=args.tau_abs,
        tau_margin=args.tau_margin,
        eval_every=args.eval_every,
        max_chunk_eval=args.max_chunk_eval,
        freeze_text=args.freeze_text,
        freeze_logit_scale=args.freeze_logit_scale,
    )

    trainer.train()


if __name__ == "__main__":
    main()
