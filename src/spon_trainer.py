"""
SPON Training Pipeline

Trains SPON biases to minimize KL divergence between dense and sparse outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging
from pathlib import Path

from .allocation import SPONConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """Training arguments for SPON."""
    epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 8
    block_size: int = 128
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    kl_temperature: float = 1.0
    warmup_steps: int = 0
    logging_steps: int = 10
    save_dir: str = "checkpoints"
    mixed_precision: bool = True
    device: str = "auto"
    # Distillation variant:
    #   - "kl": standard KL divergence (paper default)
    #   - "ce": cross-entropy to dense teacher distribution
    #   - "kl+logit_l2": KL + L2 penalty between dense and sparse logits
    distillation_loss: str = "kl"
    logit_l2_weight: float = 0.0


class SPONTrainer:
    """
    Trainer for SPON biases.
    
    Trains input-independent bias vectors to compensate for
    representation degradation from activation sparsification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SPONConfig,
        sparsity: float,
        args: TrainingArgs,
        tokenizer=None
    ):
        self.model = model
        self.config = config
        self.sparsity = sparsity
        self.args = args
        self.tokenizer = tokenizer
        from . import get_device
        self.device = get_device(args.device)
        self.device_type = self.device.type
        self.amp_enabled = args.mixed_precision and self.device_type == "cuda"
        
        # Move model to device
        self.model.to(self.device)
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Register SPON biases
        self.spon_params = self._register_spon_biases()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.spon_params,
            lr=args.learning_rate,
            weight_decay=0.0
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_enabled else None
        
        # Logging
        self.global_step = 0
        self.training_loss = []
    
    def _register_spon_biases(self) -> List[nn.Parameter]:
        """Register SPON bias parameters for layers/modules in config."""
        spon_params = []
        self.spon_bias_map = {}  # Maps layer_idx_module to parameter
        
        for layer_idx in self.config.layer_mask:
            layer = self.model.model.layers[layer_idx]
            
            for module_name in self.config.modules:
                # Get the module
                if module_name in ["down_proj", "up_proj", "gate_proj"]:
                    module = getattr(layer.mlp, module_name)
                elif module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module = getattr(layer.self_attn, module_name)
                else:
                    logger.warning(f"Unknown module: {module_name}")
                    continue
                
                out_features = module.out_features
                
                spon_bias = nn.Parameter(
                    torch.zeros(out_features, device=self.device, dtype=torch.float32)
                )
                
                key = f"layer_{layer_idx}_{module_name}"
                self.spon_bias_map[key] = spon_bias
                spon_params.append(spon_bias)
                
                logger.info(f"Registered SPON bias: {key} with dim {out_features}")
        
        logger.info(f"Total SPON parameters: {sum(p.numel() for p in spon_params)}")
        return spon_params
    
    def _get_dense_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits from dense (non-sparse) forward pass."""
        # IMPORTANT: We need to temporarily restore any modified forwards
        # to get true dense output
        self.model.eval()  # Ensure eval mode for consistent behavior
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits.detach()
    
    def _get_sparse_logits_with_spon(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get logits from sparse forward pass with SPON biases.
        
        This requires a custom forward that applies sparsification
        and adds SPON biases.
        """
        from .sparse_forward import register_sparsification_hooks, remove_hooks

        hooks = register_sparsification_hooks(
            self.model,
            sparsity=self.sparsity,
            target_modules=self.config.modules,
            spon_biases=self.spon_bias_map
        )
        try:
            outputs = self.model(input_ids)
            logits = outputs.logits
        finally:
            remove_hooks(hooks)

        return logits
    
    def _compute_distillation_loss(
        self,
        dense_logits: torch.Tensor,
        sparse_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute distillation loss between dense and sparse outputs.
        
        Supports multiple variants:
        - KL      : standard KL divergence (paper's default)
        - CE      : cross-entropy to dense teacher probabilities
        - KL+L2   : KL plus an L2 penalty on logits (logit-level matching)
        
        All computations are done in FP32 for numerical stability.
        """
        dense_logits_fp32 = dense_logits.float()
        sparse_logits_fp32 = sparse_logits.float()

        if dense_logits_fp32.dim() == 3:
            B, T, V = dense_logits_fp32.shape
            dense_logits_fp32 = dense_logits_fp32.reshape(B * T, V)
            sparse_logits_fp32 = sparse_logits_fp32.reshape(B * T, V)

        loss_type = self.args.distillation_loss.lower()

        # Teacher distribution
        teacher_probs = F.softmax(dense_logits_fp32 / temperature, dim=-1)

        # Student log-probs
        student_log_probs = F.log_softmax(sparse_logits_fp32 / temperature, dim=-1)

        if loss_type == "kl":
            base_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (temperature ** 2)
        elif loss_type == "ce":
            # Cross-entropy with soft labels: - sum p_teacher * log p_student
            ce = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
            base_loss = ce * (temperature ** 2)
        elif loss_type == "kl+logit_l2":
            kl = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (temperature ** 2)
            l2 = F.mse_loss(sparse_logits_fp32, dense_logits_fp32)
            base_loss = kl + self.args.logit_l2_weight * l2
        else:
            raise ValueError(f"Unknown distillation_loss: {self.args.distillation_loss}")

        return base_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        
        # Get dense logits (cached/no grad)
        dense_logits = self._get_dense_logits(input_ids)
        
        # Get sparse logits with SPON (requires grad)
        if self.amp_enabled:
            with torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                sparse_logits = self._get_sparse_logits_with_spon(input_ids)
                loss = self._compute_distillation_loss(
                    dense_logits,
                    sparse_logits,
                    self.args.kl_temperature
                )
        else:
            sparse_logits = self._get_sparse_logits_with_spon(input_ids)
            loss = self._compute_distillation_loss(
                dense_logits, 
                sparse_logits, 
                self.args.kl_temperature
            )
        
        scaled_loss = loss / self.args.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        return loss.item()
    
    def train(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader providing tokenized batches
        
        Returns:
            Dictionary of training metrics
        """
        # Note: We keep model in eval mode but enable gradients for SPON biases only
        # This is because we're not training the model weights, only the biases
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                leave=True
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                self.global_step += 1
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.spon_params,
                            self.args.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.spon_params,
                            self.args.max_grad_norm
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    self.training_loss.append(avg_loss)
            
            total_loss += epoch_loss
            num_steps += epoch_steps

            if epoch_steps % self.args.gradient_accumulation_steps != 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.spon_params,
                        self.args.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.spon_params,
                        self.args.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {epoch_loss / epoch_steps:.4f}")
        
        return {
            "final_loss": total_loss / num_steps,
            "training_loss": self.training_loss
        }
    
    def save_spon_biases(self, path: str):
        """Save trained SPON biases."""
        save_dict = {
            "config": {
                "name": self.config.name,
                "layer_mask": self.config.layer_mask,
                "modules": self.config.modules,
                "capacity_multiplier": self.config.capacity_multiplier
            },
            "sparsity": self.sparsity,
            "biases": {k: v.cpu() for k, v in self.spon_bias_map.items()},
            "training_args": {
                "epochs": self.args.epochs,
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.batch_size
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        logger.info(f"Saved SPON biases to {path}")
    
    def load_spon_biases(self, path: str):
        """Load trained SPON biases."""
        checkpoint = torch.load(path, map_location=self.device,weights_only=True)
        
        for key, bias in checkpoint["biases"].items():
            if key in self.spon_bias_map:
                self.spon_bias_map[key].data = bias.to(self.device)
                logger.info(f"Loaded bias: {key}")
            else:
                logger.warning(f"Bias {key} not in current config, skipping")
    
    def get_spon_biases(self) -> Dict[str, torch.Tensor]:
        """Get current SPON biases as a dictionary."""
        return {k: v.detach().clone() for k, v in self.spon_bias_map.items()}


def create_calibration_dataloader(
    tokenizer,
    block_size: int = 128,
    batch_size: int = 8,
    num_samples: int = 1024,
    dataset_name: str = "wikitext",
    dataset_subset: str = "wikitext-103-raw-v1",
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42
) -> DataLoader:
    """
    Create a DataLoader for calibration data.
    
    Args:
        tokenizer: HuggingFace tokenizer
        block_size: Token sequence length
        batch_size: Batch size
        num_samples: Number of calibration samples
        dataset_name: Dataset name
        dataset_subset: Dataset subset
        split: Dataset split ("train", "validation", or "test")
        shuffle: Whether to shuffle the dataloader
        seed: Random seed for reproducible shuffling
    
    Returns:
        DataLoader for calibration
    """
    from datasets import load_dataset

    # Backward-compatible normalization for old WikiText subset names.
    alias_map = {
        "wikitext-raw-v2": "wikitext-2-raw-v1",
        "wikitext-2-raw-v2": "wikitext-2-raw-v1",
        "wikitext-103-raw-v2": "wikitext-103-raw-v1",
    }
    requested_subset = dataset_subset
    normalized_subset = alias_map.get(dataset_subset, dataset_subset)

    dataset = None
    try:
        dataset = load_dataset(dataset_name, normalized_subset, split=split)
    except ValueError as e:
        if dataset_name != "wikitext":
            raise

        fallback_subsets = [
            "wikitext-2-raw-v1",
            "wikitext-103-raw-v1",
            "wikitext-2-v1",
            "wikitext-103-v1",
        ]
        for candidate in fallback_subsets:
            if candidate == normalized_subset:
                continue
            try:
                dataset = load_dataset(dataset_name, candidate, split=split)
                logger.warning(
                    "Dataset subset '%s' unavailable for %s; falling back to '%s'.",
                    requested_subset,
                    dataset_name,
                    candidate,
                )
                break
            except Exception:
                continue

        if dataset is None:
            raise ValueError(
                f"Failed to load dataset '{dataset_name}' with subset '{requested_subset}' "
                f"(normalized='{normalized_subset}') for split '{split}'."
            ) from e
    
    all_token_ids = []
    target_tokens = num_samples * block_size
    
    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False
        )["input_ids"][0]
        all_token_ids.append(tokens)
        if sum(t.numel() for t in all_token_ids) >= target_tokens + block_size:
            break
    
    all_tokens = torch.cat(all_token_ids)
    
    chunks = []
    for i in range(0, len(all_tokens) - block_size, block_size):
        chunks.append(all_tokens[i:i + block_size])
        if len(chunks) >= num_samples:
            break
    
    if len(chunks) == 0:
        raise ValueError(
            f"No samples created from split='{split}'. Check if dataset has "
            f"enough text. Need at least {block_size} tokens per chunk."
        )
    
    input_ids = torch.stack(chunks)
    tensor_dataset = torch.utils.data.TensorDataset(input_ids)
    
    def collate_fn(batch):
        return {"input_ids": torch.stack([item[0] for item in batch])}
    
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator
    )
    
    return dataloader
