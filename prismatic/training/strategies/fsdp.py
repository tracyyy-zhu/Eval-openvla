"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

import os
import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
import contextlib
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class FSDPStrategy(TrainingStrategy):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        lr_num_cycles: float = 0.5,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP Sharding Strategy {sharding_strategy} is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        self.lr_num_cycles = lr_num_cycles
        self.stage = stage

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        # assert isinstance(self.vlm, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with contextlib.ExitStack() as stack:
            if isinstance(self.vlm, FSDP):
                stack.enter_context(FSDP.state_dict_type(self.vlm, StateDictType.FULL_STATE_DICT, full_cfg))
            for m in self.vlm.modules():
                if isinstance(m, FSDP) and m is not self.vlm:
                    stack.enter_context(FSDP.state_dict_type(m, StateDictType.FULL_STATE_DICT, full_cfg))
            full_vlm_state_dict = self.vlm.state_dict()

        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            # full_vlm_state_dict = self.vlm.state_dict()
            model_state_dicts = {
                mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
            }

            # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            for key, param in full_vlm_state_dict.items():
                for mkey in model_state_dicts:
                    if key.startswith(mprefix := f"{mkey}."):
                        model_state_dicts[mkey][key.removeprefix(mprefix)] = param

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / "checkpoints"
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f"SigLIP-step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt" #flag
                else:
                    checkpoint_path = (
                        checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                    )

                # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                print("checkpoint_path", checkpoint_path)
                # checkpoint_path /scratch/yz5880/openvla/checkpoints/single_projector/DINO/OpenVLA_train/checkpoints/step-000002-epoch-00-loss=10.8889.pt
                torch.save({"model": model_state_dicts}, checkpoint_path)


                # TODO (siddk) :: This breaks w/ Sagemaker default permissions (root vs. <user>)... skip?
                # shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def unwrap_all(self, m):
        while isinstance(m, FSDP): m = m.module
        return m

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        # vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()
        vlm_fsdp_wrapping_policy = self.vlm.get_projector_wrapping_policy() # Wrap only projector

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
            #   => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype
            )

            # When running FSDP with a frozen vision backbone --> move to half precision!
            if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
                overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)

        else:
            # If we're not using mixed precision, everything is in default full precision!
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        # base  = self.unwrap_all(self.vlm)
        # llm   = self.unwrap_all(base.llm_backbone)
        # llama = getattr(llm, "llm", llm)

        # emb = llama.get_input_embeddings().weight
        # head = llama.get_output_embeddings().weight
        # assert emb.ndim == 2 and head.ndim == 2, (emb.shape, head.shape)

        if self.stage == "align_projector":
            self.vlm.projector = FSDP(
                self.vlm.projector, 
                auto_wrap_policy=vlm_fsdp_wrapping_policy,
                mixed_precision=fsdp_precision_policy,
                sharding_strategy=self.fsdp_sharding_strategy,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                use_orig_params=True)
        elif self.stage == "align_vision_projector":
            # Wrap BOTH the vision backbone and the projector, but not the whole VLM.
            if not isinstance(self.vlm.vision_backbone, FSDP):
                self.vlm.vision_backbone = FSDP(
                    self.vlm.vision_backbone,
                    auto_wrap_policy=vlm_fsdp_wrapping_policy,
                    mixed_precision=fsdp_precision_policy,
                    sharding_strategy=self.fsdp_sharding_strategy,
                    device_id=torch.cuda.current_device(),
                    limit_all_gathers=True,
                    use_orig_params=True,
                )
            if not isinstance(self.vlm.projector, FSDP):
                self.vlm.projector = FSDP(
                    self.vlm.projector, 
                    auto_wrap_policy=vlm_fsdp_wrapping_policy,
                    mixed_precision=fsdp_precision_policy,
                    sharding_strategy=self.fsdp_sharding_strategy,
                    device_id=torch.cuda.current_device(),
                    limit_all_gathers=True,
                    use_orig_params=True,
                )
        else:
            self.vlm = FSDP(
                self.vlm,
                auto_wrap_policy=vlm_fsdp_wrapping_policy,
                mixed_precision=fsdp_precision_policy,
                sharding_strategy=self.fsdp_sharding_strategy,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                use_orig_params=True,
            )
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        self.vlm.to(device)

        def assert_same_device(*mods):
            devs = []
            for m in mods:
                t = next(m.parameters(), None)
                if t is None:
                    t = next(m.buffers(), None)
                if t is None:
                    continue  # skip paramless
                devs.append(t.device)
            assert len(set(devs)) <= 1, f"Device mismatch: {devs}"

        assert_same_device(self.vlm.vision_backbone, self.vlm.llm_backbone, self.vlm.projector.module)

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == "align_cosine_decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            print("self.warmup_ratio", self.warmup_ratio)
            print("num_warmup_steps", num_warmup_steps)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            if self.stage == "align_vision_projector":
                proj_params, bb_block23, bb_block22, bb_final_norms, other = [], [], [], [], []

                for n, p in self.vlm.named_parameters():  # model contains projector + dino_featurizer
                    if not p.requires_grad:
                        continue
                    if "projector" in n:
                        proj_params.append(p)
                    elif "dino_featurizer" in n:
                        if ".blocks.23." in n:
                            bb_block23.append((n, p))
                        elif ".blocks.22." in n:
                            bb_block22.append((n, p))
                        elif n.endswith("dino_featurizer.norm.weight") or n.endswith("dino_featurizer.norm.bias") \
                            or ".vit.norm." in n:  # depending on wrapper
                            bb_final_norms.append((n, p))
                        else:
                            other.append((n, p))  # should be empty if you froze correctly
                def is_norm_or_bias(n, p):
                    return p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower()
                def pg_from(named_params, lr):
                    wd, no_wd = [], []
                    for n, p in named_params:
                        (no_wd if is_norm_or_bias(n, p) else wd).append(p)
                    groups = []
                    if wd: groups.append({"params": wd, "lr": lr, "weight_decay": 0.05})
                    if no_wd: groups.append({"params": no_wd, "lr": lr, "weight_decay": 0.0})
                    return groups
                
                # LRs
                lr_proj = self.learning_rate
                print("self.learning_rate", self.learning_rate)
                lr_block23 = self.learning_rate / 30         # highest among backbone groups
                lr_block22 = lr_block23 * 0.7        # decay a bit
                lr_final_norms = lr_block22 * 0.7     # small but nonzero
                
                param_groups = []
                param_groups += [{"params": proj_params, "lr": lr_proj, "weight_decay": 0.05}]
                param_groups += pg_from(bb_block23, lr_block23)
                param_groups += pg_from(bb_block22, lr_block22)
                param_groups += pg_from(bb_final_norms, lr_final_norms)

                self.optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)

            elif self.stage == "align_projector":                
                decay, no_decay = [], []
                for name, param in self.vlm.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                # Build Parameter Groups
                param_groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

                self.optimizer = AdamW(param_groups, lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-8)

            # Create Optimizer & LR Scheduler
            # self.optimizer = AdamW(param_groups, lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-8)
            # self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps) 
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps, num_cycles=self.lr_num_cycles) # num_cycles use 2 or 3
            self.print_lr_by_prefix(self.vlm, self.optimizer, prefixes=("projector","vision_backbone","dino_featurizer","siglip_featurizer"))

            # LR range test -- MAKE SURE YOUR STEPS IS NOT TOO LARGE OVER 1000
            # print("LR range test ...")
            # self.optimizer = AdamW(groups, lr=2e-6, betas=(0.9, 0.98), eps=1e-8)
            # lr_start, lr_end = 2e-7, 5e-2
            # T = 1000 # !!! MAX STEPS
            # def lr_lambda(step):
            #     r = lr_end / lr_start
            #     return (r ** (step / max(1, T-1)))
            # for pg in self.optimizer.param_groups:
            #     pg['lr'] = lr_start
            # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        # self.max_grad_norm == 1.0
        # self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)
        # self.vlm.projector.clip_grad_norm_(max_norm=self.max_grad_norm)
        if FSDP is not None and isinstance(self.vlm, FSDP):
            norm_type = 2.0
            total_norm = FSDP.clip_grad_norm_(self.vlm, self.max_grad_norm, norm_type=norm_type)
            return float(total_norm)

    def print_lr_by_prefix(self, model, optimizer, prefixes=("projector", "vision_backbone", "dino_featurizer", "siglip_featurizer")):
        # map param id -> lr of its param_group
        pid2lr = {}
        for pg in optimizer.param_groups:
            lr = pg.get("lr", optimizer.defaults.get("lr"))
            for p in pg["params"]:
                pid2lr[id(p)] = lr

        # aggregate LRs by module prefix
        from collections import defaultdict
        seen = {pref: set() for pref in prefixes}
        counts = defaultdict(int)

        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            for pref in prefixes:
                if name.startswith(pref):
                    lr = pid2lr.get(id(p), None)
                    if lr is not None:
                        seen[pref].add(lr)
                        counts[(pref, lr)] += 1
                    break

        # pretty print
        print("\n[LR CHECK]")
        for pref in prefixes:
            if seen[pref]:
                lrs = sorted(seen[pref])
                print(f"  {pref}: unique LRs = {lrs}")
                for lr in lrs:
                    print(f"    - {counts[(pref, lr)]} params @ lr={lr}")
            else:
                print(f"  {pref}: (no trainable params found)")
        print("")
