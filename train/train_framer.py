#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import shutil
import sys
sys.path.append(os.getcwd())
from pathlib import Path

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from model_framer.transformer_sd3 import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

import torch.distributed as dist
import torch.nn.functional as F
import random

def generate_frequency_masks(fft_tensor, low_freq_percent=0.2):
    """
    Generate low and high frequency masks from FFT tensor.
    
    Decomposes frequency domain into low and high frequency components based on
    magnitude quantile threshold.
    
    Args:
        fft_tensor: FFT-transformed tensor of shape [B, C, H, W] or [H, W]
        low_freq_percent: Percentile threshold (0-1) for low frequency cutoff.
                            Components below this percentile are considered low-frequency.
    
    Returns:
        low_masks_batch: Binary mask for low frequency components [B, C, H, W]
        high_masks_batch: Binary mask for high frequency components [B, C, H, W]
    """
    original_dim = fft_tensor.dim()
    if original_dim < 2:
        raise ValueError("The input tensor must be at least 2-dimensional.")
    if original_dim == 2:  # [H, W] -> [1, H, W]
        fft_tensor = fft_tensor.unsqueeze(0)

    magnitudes = torch.abs(fft_tensor)
    flat_magnitudes = magnitudes.view(magnitudes.shape[0], -1)
    cutoff_values = torch.quantile(flat_magnitudes, low_freq_percent, dim=1)
    cutoff_values_reshaped = cutoff_values.view(-1, *([1] * (magnitudes.dim() - 1)))

    low_masks_batch = (magnitudes < cutoff_values_reshaped).float()
    high_masks_batch = (magnitudes >= cutoff_values_reshaped).float()

    if original_dim == 2:
        low_masks_batch = low_masks_batch.squeeze(0)
        high_masks_batch = high_masks_batch.squeeze(0)

    return low_masks_batch, high_masks_batch

def intra_loss(
    q_norm: torch.Tensor, 
    k_norm: torch.Tensor, 
    negative_norm: torch.Tensor,
    tau: float, 
    reduction: str = 'mean'
):
    """
    Intra Contrastive Loss (IntraCL).
    
    Compares positive pair (query, positive_key) against negative sample within
    the same frequency band. This preserves local structure and consistency within
    frequency components.
    
    Args:
        q_norm: Normalized query features [B, D] from current student layer
        k_norm: Normalized positive key features [B, D] from teacher representation
        negative_norm: Normalized negative features [B, D] from different student layer
        tau: Temperature parameter for softmax scaling
        reduction: Loss reduction method ('mean' or 'none')
    
    Returns:
        loss: Contrastive loss value (scalar or per-sample)
    """
    device = q_norm.device
    B = q_norm.shape[0]

    # Compute cosine similarity scores: s_ij = q_i · k_j
    pos_sim = (q_norm * k_norm).sum(dim=1)
    neg_sim = (q_norm * negative_norm).sum(dim=1)

    # Logits for binary classification: [pos_sim, neg_sim]
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    logits = logits / tau

    # Target label is always 0 (positive sample should have highest probability)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    
    return loss

def inter_loss(
    q_norm: torch.Tensor, 
    k_norm: torch.Tensor, 
    hard_k_norm: torch.Tensor,
    tau: float, 
    accelerator: Accelerator,
    reduction: str = 'mean'
):
    """
    Inter Contrastive Loss (InterCL).
    
    Compares query against all negative keys across the batch and distributed
    processes. Hard negatives are selected from different student layers to
    encourage discriminative learning of high-frequency details.
    
    In distributed setting, gathers representations from all processes to create
    a large negative pool, improving contrastive learning effectiveness.
    
    Args:
        q_norm: Normalized query features [B, D] from current student layer
        k_norm: Normalized positive key features [B, D] from teacher
        hard_k_norm: Hard negative samples [B, D] from different student layer.
                    Selected to maximize disagreement with positive representations.
        tau: Temperature parameter for softmax scaling
        accelerator: Accelerator instance for distributed gathering (multi-GPU/multi-node)
        reduction: Loss reduction method ('mean' or 'none')
    
    Returns:
        loss: Cross-entropy loss over gathered negatives across all processes
    """
    B = q_norm.shape[0]
    device = q_norm.device

    # Gather positive keys from all processes for distributed training
    if accelerator.num_processes > 1:
        k_norm_all = accelerator.gather(k_norm.contiguous())
    else:
        k_norm_all = k_norm

    # Combine positive keys with hard negatives for full negative pool
    if hard_k_norm is not None:
        if accelerator.num_processes > 1:
            hard_k_norm_all = accelerator.gather(hard_k_norm).detach()
        else:
            hard_k_norm_all = hard_k_norm.detach()
        
        k_final = torch.cat([k_norm_all, hard_k_norm_all], dim=0)
    else:
        k_final = k_norm_all

    # Compute logits: [B, B*num_processes] for gathered batch size
    logits = (q_norm @ k_final.T) / tau
    
    # Labels account for process index: each sample's positive is at 
    # index = process_index * B + local_index in the gathered tensor
    labels = torch.arange(B, device=device) + accelerator.process_index * B
    
    return F.cross_entropy(logits, labels, reduction=reduction)

def framer_distillation_loss(
    student_hidden_states: list,
    teacher_hidden_state: torch.Tensor,
    noisy_model_input: torch.Tensor,
    sigmas: torch.Tensor,
    accelerator,
    *,
    cutoff: float = 0.2,
    tau_high: float = 1.0
) -> torch.Tensor:
    """
    Frequency-Aligned Self-Distillation Loss (FRAMER) on Raw Hidden States.
    
    Implements hierarchical contrastive distillation with adaptive modulation by
    decomposing the spatial frequency domain of intermediate transformer features 
    into low and high-frequency components.
    
    Key insight: 
    - Low frequencies of hidden states encode structural layout → preserve via IntraCL
    - High frequencies of hidden states encode fine textural details → preserve via InterCL
    
    Loss computation:
    1. Directly transform raw teacher and student spatial features (B, C, H, W) into FFT domain.
    2. Apply frequency masks to separate low/high components.
    3. Compute energy distribution per frequency band.
    4. Weight losses by: 
       (a) Intensity (Cosine similarity to teacher's frequency components).
       (b) Dynamic per-sample energy distribution divergence between student and teacher.
    5. Average across all student intermediate layers.
    
    Math Formulation:
    - Student features: q_i = f_i(x_t) where f_i is the output of the i-th transformer block.
    - Teacher feature: k = f_n(x_t) where f_n is the final transformer block output.
    - FFT decomposition: F_low = F · M_low, F_high = F · M_high
    - Energy: E_band = Σ|F_band| / |support(M_band)|
    - Dynamic weight: w_dynamic = 1 / (1 + |ΔE|) where ΔE = |E_teacher - E_student| / E_student
    - Final loss: L = Σ_i (w_low * L_intra_low + w_high * L_inter_high)
    
    Args:
        student_hidden_states (list of torch.Tensor): 
            List of intermediate student layer features of shape [B, C, H, W].
            Unpatchified directly from the sequence without projection heads.
        teacher_hidden_state (torch.Tensor): 
            Final teacher feature representation of shape [B, C, H, W].
            Should be from the frozen teacher network, unpatchified.
        accelerator (Accelerator): 
            Distributed training accelerator (supports multi-GPU/node) used for 
            gathering representations across processes for inter_loss.
        cutoff (float, optional): 
            Frequency percentile (0-1) threshold for low/high decomposition.
            Defaults to 0.2 (bottom 20% magnitude frequencies are "low").
        tau_high (float, optional): 
            Temperature parameter for contrastive loss softmax scaling. Defaults to 1.0.
    
    Returns:
        total_distillation_loss (torch.Tensor): 
            Scalar loss tensor computed as the mean of per-layer adaptively 
            weighted contrastive losses.
            
    Implementation Details:
    - Bypasses final projection layers (norm/proj) to prevent structural distortion 
      caused by domain mismatch between intermediate and final feature distributions.
    - Applies FFT directly across the spatial dimensions (H, W) of the hidden channels.
    """
    
    # Extract dimensions directly from the teacher's hidden state
    B, C, H, W = teacher_hidden_state.shape
    device = teacher_hidden_state.device
    positive_latent_k = teacher_hidden_state * (-sigmas) + noisy_model_input
    
    # 1. Direct FFT transform of the teacher feature (No denoise reconstruction)
    k_fft = torch.fft.fftshift(torch.fft.fft2(positive_latent_k.float()), dim=(-2, -1))
    teacher_mag = torch.abs(k_fft)

    layer_losses = []
    
    layer_w_low = []
    layer_w_high = []
    
    # Frequency domain decomposition: separate low and high frequency masks
    low_mask, high_mask = generate_frequency_masks(k_fft, cutoff)
    
    eps = 1e-9
    
    # Extract frequency components: element-wise multiplication with masks
    k_low_freq_mag = torch.abs(k_fft * low_mask)
    k_high_freq_mag = torch.abs(k_fft * high_mask)
    
    # Flatten spatial dimensions for contrastive learning: [B, C*H*W]
    k_low_flat = k_low_freq_mag.view(B, -1)
    k_high_flat = k_high_freq_mag.view(B, -1)
    
    # L2 normalize for fair cosine similarity in contrastive objectives
    k_low_flat = F.normalize(k_low_flat, p=2, dim=1).detach()
    k_high_flat = F.normalize(k_high_flat, p=2, dim=1).detach()
    
    # === Process each student intermediate layer ===
    for i in range(len(student_hidden_states)):
        # Current student layer spatial feature
        current_feat = student_hidden_states[i] * (-sigmas) + noisy_model_input

        # Hard negative: sample from a different student layer
        # Encourages student layers to learn diverse, non-collapsing representations
        available_indices = list(range(len(student_hidden_states)))
        available_indices.remove(i)
        rand_idx = random.choice(available_indices)
        # prev_feat = student_hidden_states[rand_idx]
        prev_feat = student_hidden_states[rand_idx] * (-sigmas) + noisy_model_input

        # FFT transform of student features directly
        current_fft = torch.fft.fftshift(torch.fft.fft2(current_feat.float()), dim=(-2, -1))
        prev_fft = torch.fft.fftshift(torch.fft.fft2(prev_feat.float()), dim=(-2, -1))
        current_mag = torch.abs(current_fft)
        
        # === Compute energy distribution per frequency band ===
        den_low  = low_mask.sum(dim=(-2, -1), keepdim=True) + eps
        den_high = high_mask.sum(dim=(-2, -1), keepdim=True) + eps

        # Energy in each frequency band: E_band = Σ|F_band| / |support(M_band)|
        E_i_low  = (current_mag * low_mask).sum(dim=(-2, -1), keepdim=True)  / den_low
        E_i_high = (current_mag * high_mask).sum(dim=(-2, -1), keepdim=True) / den_high

        E_n_low  = (teacher_mag * low_mask).sum(dim=(-2, -1), keepdim=True)  / den_low
        E_n_high = (teacher_mag * high_mask).sum(dim=(-2, -1), keepdim=True) / den_high

        # Average over channels: [B, C, 1, 1] → [B, 1, 1]
        E_i_low  = E_i_low.mean(dim=1)
        E_i_high = E_i_high.mean(dim=1)
        E_n_low  = E_n_low.mean(dim=1)
        E_n_high = E_n_high.mean(dim=1)

        # === Dynamic weights based on energy divergence ===
        # ΔE = |E_teacher - E_student| / E_student (relative error)
        Delta_low  = (E_n_low  - E_i_low ).abs() / (E_i_low  + eps)
        Delta_high = (E_n_high - E_i_high).abs() / (E_i_high + eps)

        # Weight function: ranges from ~1 (perfect match) to ~0 (large divergence)
        sim_mag_low  = 1.0 / (1.0 + Delta_low)
        sim_mag_high = 1.0 / (1.0 + Delta_high)

        # Squeeze to shape [B]
        w_mag_low  = sim_mag_low.squeeze(-1).squeeze(-1).detach()
        w_mag_high = sim_mag_high.squeeze(-1).squeeze(-1).detach()
        
        faw_high = w_mag_low
        faw_low = w_mag_high
        
        # Extract frequency components from current and hard-negative student layers
        q_low_freq_mag = torch.abs(current_fft * low_mask)
        q_high_freq_mag = torch.abs(current_fft * high_mask)
        
        hard_k_low_freq_mag = torch.abs(prev_fft * low_mask)
        hard_k_high_freq_mag = torch.abs(prev_fft * high_mask)

        # Flatten for contrastive learning
        q_low_flat = q_low_freq_mag.view(B, -1)
        q_high_flat = q_high_freq_mag.view(B, -1)
        
        hard_k_low_flat = hard_k_low_freq_mag.view(B, -1)
        hard_k_high_flat = hard_k_high_freq_mag.view(B, -1)       
        
        # L2 normalize for cosine similarity-based contrastive objectives
        q_low_flat = F.normalize(q_low_flat, p=2, dim=1)
        q_high_flat = F.normalize(q_high_flat, p=2, dim=1)
        hard_k_low_flat = F.normalize(hard_k_low_flat, p=2, dim=1).detach()
        hard_k_high_flat = F.normalize(hard_k_high_flat, p=2, dim=1).detach()

        # === Intensity weights: cosine similarity to teacher frequency components ===
        sim_low = F.relu(F.cosine_similarity(q_low_flat, k_low_flat, dim=1))
        sim_high = F.relu(F.cosine_similarity(q_high_flat, k_high_flat, dim=1))
        
        fam_low = sim_low.detach()
        fam_high = sim_high.detach()

        # === Compute contrastive losses per frequency band ===
        loss_low_per_sample = intra_loss(q_low_flat, k_low_flat, hard_k_low_flat, tau_high, reduction='none')            
        loss_high_per_sample = inter_loss(q_high_flat, k_high_flat, hard_k_high_flat, tau_high, accelerator, reduction='none')
                
        # === Final adaptive weighting ===
        final_w_low = fam_low * faw_high
        final_w_high = fam_high * faw_low
        layer_w_low.append(final_w_low.mean())
        layer_w_high.append(final_w_high.mean())
        
        # Weighted loss: apply adaptive weights and combine frequency bands
        weighted_loss_per_sample = (final_w_low * loss_low_per_sample) + (final_w_high * loss_high_per_sample)
        layer_loss = weighted_loss_per_sample.mean()
        layer_losses.append(layer_loss)

    # Average distillation loss across all student intermediate layers
    total_distillation_loss = torch.mean(torch.stack(layer_losses))
    avg_w_low = torch.mean(torch.stack(layer_w_low))
    avg_w_high = torch.mean(torch.stack(layer_w_high))
    
    return total_distillation_loss, avg_w_low, avg_w_high


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='NOTHING',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])

    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.2,
        help=(
            "Cutoff value for hierarchical contrastive distillation."
        ),
    )
    parser.add_argument(
        "--tau_high",
        type=float,
        default=1.0,
        help=(
            "Temperature parameter for high-level contrastive distillation."
        ),
    )
    parser.add_argument(
        "--lambda_framer",
        type=float,
        default=0.1,
        help=(
            "Weight for the FRAMER distillation loss in the overall training objective."
        ),
    )

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args



# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    # prompt_embeds = clip_prompt_embeds

    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if args.transformer_model_name_or_path is not None:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    else:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    transformer.requires_grad_(False)

    # release the cross-attention part in the unet.
    for name, params in transformer.named_parameters():
        # if name.endswith(tuple(args.trainable_modules)):
        if any(trainable_modules in name for trainable_modules in tuple(args.trainable_modules)):
            print(f'{name} in <transformer> will be optimized.' )
            # for params in module.parameters():
            params.requires_grad = True
    

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "transformer"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            # while len(models) > 0:
                # pop models so that they are not loaded again
            model = models.pop()

            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True


            # load diffusers style into model
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # params_to_optimize = controlnet.parameters()
    # params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    # params_to_optimize = [transformer_parameters_with_lr]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    transformer.to(accelerator.device, dtype=weight_dtype)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    train_dataset = PairedCaptionDataset(root_folder=args.root_folders, 
                                         null_text_ratio=args.null_text_ratio,
)

    def compute_text_embeddings(batch, text_encoders, tokenizers):
        with torch.no_grad():
            prompt = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    free_memory()

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                model_input = batch["pixel_values"].to(dtype=weight_dtype)
                # with torch.no_grad():
                #     model_input = vae.encode(pixel_values).latent_dist.sample()
                #     model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                # controlnet(s) inference
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                # with torch.no_grad():
                #     controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
                #     controlnet_image = (controlnet_image - vae.config.shift_factor)  * vae.config.scaling_factor
                # image_embedding = controlnet_image.view(controlnet_image.shape[0], 16, -1)
                # pad_tensor = torch.zeros(controlnet_image.shape[0], 77 - image_embedding.shape[1], 4096).to(image_embedding.device, dtype=weight_dtype)
                # image_embedding = torch.cat([image_embedding, pad_tensor], dim=1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                # input_model_input = torch.cat([noisy_model_input, controlnet_image], dim = 1)

                # Get the text embedding for conditioning
                # prompts = compute_text_embeddings(batch, text_encoders, tokenizers)
                prompt_embeds = batch["prompt_embeds"].to(dtype=model_input.dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=model_input.dtype)
                # prompt_embeds = torch.cat([prompt_embeds, image_embedding], dim=-2)

                # Predict the noise residual
                model_pred, pred_hidden_states = transformer(
                    hidden_states=noisy_model_input,
                    controlnet_image=controlnet_image,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )
                student_features = pred_hidden_states[:-1]
                teacher_feature = pred_hidden_states[-1].detach()

                distillation_loss, _, _ = framer_distillation_loss(
                    student_hidden_states=student_features,
                    teacher_hidden_state=teacher_feature,
                    accelerator=accelerator,
                    cutoff = args.cutoff,
                    tau_high=args.tau_high
                )

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean() + distillation_loss * args.lambda_framer

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # params_to_clip = controlnet.parameters()
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    #     image_logs = log_validation(
                    #         transformer,
                    #         args,
                    #         accelerator,
                    #         weight_dtype,
                    #         global_step,
                    #     )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
