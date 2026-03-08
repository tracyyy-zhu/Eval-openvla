"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
from safetensors.torch import load_file as safe_load_file

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft.tuners.lora import LoraLayer
from transformers import get_cosine_schedule_with_warmup

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    merge_dir: str = None                                           # Optional path to merged LoRA checkpoint in initialization (DINOv3)

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        print("Using LoRA fine-tuning with rank", cfg.lora_rank, "and dropout", cfg.lora_dropout)
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True) 
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,  # False avoids meta tensors
        # device_map={"": "cuda:0"}, # Used with low_cpu_mem_usage=True {"": device_id},  # or "auto" if you want automatic placement
        trust_remote_code=True,
    )
    if cfg.merge_dir is not None:
        vla = PeftModel.from_pretrained(vla, cfg.merge_dir).merge_and_unload()
    import glob
    # Find all shards (00001, 00002, 00003)
    shard_files = glob.glob(f"{cfg.vla_path}/pytorch_model-*.bin")
    for shard in shard_files:
        print(f"Manually healing weights from {shard}...")
        sd = torch.load(shard, map_location="cpu", mmap=True, weights_only=True)
        
        # We only care about the keys that were 'junked'
        # This specifically targets the vision backbone gammas
        # Force the model to take these healthy values
        fix_dict = {}
        for k, v in sd.items ():
            if "gamma" in k or "layer_scale" in k:
                # Move only these tiny vectors to the GPU
                fix_dict[k] = v. to(dtype=torch.bfloat16)
        if fix_dict:
            print(f" → Fixing {len(fix_dict)} gamma parameters...") 
            vla.load_state_dict(fix_dict, strict=False)
        
        # Force the model to take these healthy values
        del sd
        del fix_dict
    # print("Check after proper load:", vla.vision_backbone.featurizer.vit.blocks[10].gamma_1[:5])
    def load_old_lora_weights_into_expanded_model(peft_model, old_adapter_dir):
        """
        Load an old adapter checkpoint into a newly created expanded LoRA model.
        Overlapping LoRA tensors are restored.
        Newly added LoRA layers stay randomly initialized.
        """
        adapter_safetensors = os.path.join(old_adapter_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(old_adapter_dir, "adapter_model.bin")

        if os.path.exists(adapter_safetensors):
            old_sd = safe_load_file(adapter_safetensors)
        elif os.path.exists(adapter_bin):
            old_sd = torch.load(adapter_bin, map_location="cpu")
        else:
            raise FileNotFoundError(f"Cannot find adapter weights in {old_adapter_dir}")

        missing, unexpected = peft_model.load_state_dict(old_sd, strict=False)
        print(f"Loaded old adapter weights from {old_adapter_dir}")
        print(f"Missing keys count: {len(missing)}")
        print(f"Unexpected keys count: {len(unexpected)}")

        return peft_model

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    vla.requires_grad_(False)
    if cfg.use_lora:
        # Dynamically find the exact module paths
        target_modules = []
        for name, module in vla.named_modules():
            # Only target Linear layers (LoRA requirement)
            if isinstance(module, torch.nn.Linear):
                # Match Vision Backbone layers (including the .vit. part)
                if "vision_backbone.featurizer" in name and any(k in name for k in ["patch_embed.proj", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]):
                    target_modules.append(name)
                # Match Projector layers
                elif "projector" in name and any(k in name for k in ["fc1", "fc2", "fc3"]):
                    target_modules.append(name)
                elif "language_model" in name and any(k in name for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]):
                    target_modules.append(name)

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16), # recommended to keep lora_alpha:lora_rank=1:1
            lora_dropout=cfg.lora_dropout,
            # target_modules="all-linear",
            target_modules=target_modules, # Use the list we just built
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        # if cfg.merge_dir is not None:
        #     vla = load_old_lora_weights_into_expanded_model(vla, cfg.merge_dir)

        vla.print_trainable_parameters()
        # for name, module in vla.named_modules():
        #     if isinstance(module, LoraLayer):
        #         print("LoraLayer:", name, type(module))

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True,  gradient_as_bucket_view=True) 

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # # LR range test
    # print("In the process of LR range test ...")
    # lr_start, lr_end = 1e-7, 5e-2
    # num_steps = 1000 # !!! MAX STEPS
    # lr_factor = (lr_end / lr_start) ** (1 / num_steps)
    # for pg in optimizer.param_groups:
    #     pg['lr'] = lr_start

    # Calculate steps
    total_steps = cfg.max_steps
    warmup_steps = int(0.03 * total_steps) # 3% warmup is standard

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss
            assert torch.isfinite(loss).item(), f"loss is not finite: {loss.detach().float().item()}\n"

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            with torch.no_grad():
                logits = output.logits.detach()
                action_logits = logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress.update()

                # # Update LR for LR Range Test
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] *= lr_factor
                    
                # Optional: Log the current LR to WandB to see it against the loss
                if distributed_state.is_main_process:
                    wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=gradient_step_idx)
                # progress.update()
                
                # # Stop early once the LR range test is done
                # if gradient_step_idx >= num_steps:
                #     print("LR Range Test Complete. Check WandB for the 'elbow' in the loss curve.")
                #     return

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir, safe_serialization=True)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    # base_vla = AutoModelForVision2Seq.from_pretrained(
                    #     cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=True
                    # ).to(device_id) # This creates meta tensor
                    peft_model = vla.module if hasattr(vla, "module") else vla

                    # merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    # merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir, safe_serialization=True)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            import torch.nn as nn

                            def force_unique_gammas(model):
                                vit = model.vision_backbone.featurizer.vit
                                for blk in vit.blocks:
                                    if hasattr(blk, "gamma_1") and blk.gamma_1 is not None:
                                        new_p = nn.Parameter(blk.gamma_1.detach().clone().contiguous(),
                                                            requires_grad=blk.gamma_1.requires_grad)
                                        blk._parameters["gamma_1"] = new_p
                                    if hasattr(blk, "gamma_2") and blk.gamma_2 is not None:
                                        new_p = nn.Parameter(blk.gamma_2.detach().clone().contiguous(),
                                                            requires_grad=blk.gamma_2.requires_grad)
                                        blk._parameters["gamma_2"] = new_p

                            def assert_gammas_unshared(model):
                                vit = model.vision_backbone.featurizer.vit
                                seen = set()
                                for i, blk in enumerate(vit.blocks):
                                    for name in ["gamma_1", "gamma_2"]:
                                        if hasattr(blk, name) and getattr(blk, name) is not None:
                                            p = getattr(blk, name)
                                            key = (name, int(p.data_ptr()))
                                            if key in seen:
                                                raise RuntimeError(f"still shared: blocks.{i}.{name} has duplicate data_ptr {p.data_ptr()}")
                                            seen.add(key)

                            # merged_vla.to("cpu")
                            # vit = merged_vla.vision_backbone.featurizer.vit
                            # for i, blk in enumerate(vit.blocks[:5]):
                            #     for n in ["gamma_1", "gamma_2"]:
                            #         if hasattr(blk, n) and getattr(blk, n) is not None:
                            #             p = getattr(blk, n)
                            #             print(
                            #                 i, n,
                            #                 "shape", tuple(p.shape),
                            #                 "numel", p.numel(),
                            #                 "device", p.device,
                            #                 "dtype", p.dtype,
                            #                 "data_ptr", int(p.data_ptr()),
                            #                 "is_meta", (p.device.type == "meta"),
                            #             )
                            # force_unique_gammas(merged_vla)
                            # assert_gammas_unshared(peft_model)

                            processor.save_pretrained(checkpoint_dir, safe_serialization=True)
                            peft_model.save_pretrained(checkpoint_dir, safe_serialization=True)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
