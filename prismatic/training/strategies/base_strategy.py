"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        val_interval: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size
        self.val_interval = val_interval

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        # val_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )
        print("len(vla_dataset)", len(vla_dataset))
        print("len(dataloader)", len(dataloader))
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=self.per_device_batch_size,
        #     sampler=None,
        #     collate_fn=collator,
        #     num_workers=0,
        #     worker_init_fn=self.worker_init_fn,
        # )
        # print("[DEBUG] val_dataset object:", val_dataset)
        # print("[DEBUG] hasattr(val_dataset, '__iter__'):", hasattr(val_dataset, "__iter__"))

        # count = 0
        # for i, sample in enumerate(val_dataset):
        #     print("[DEBUG] first raw sample keys:", sample.keys())
        #     count += 1
        #     if i == 2:
        #         break

        # print("[DEBUG] val_dataset yielded", count, "samples in raw iteration")

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad(set_to_none=True)

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            # n_step = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                # print(f"=============== STEP {n_step} ====================")
                # n_step += 1                
                 
                batch = self.move_to_device(batch, device)

                with torch.autocast(
                    "cuda", 
                    dtype=self.mixed_precision_dtype, 
                    enabled=self.enable_mixed_precision_training,
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
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

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                def check_grads_none(model):
                    # Returns (num_params, num_none, num_tensor)
                    num_params = num_none = num_tensor = 0
                    for p in model.parameters():
                        if not p.requires_grad: 
                            continue
                        num_params += 1
                        if p.grad is None:
                            num_none += 1
                        else:
                            num_tensor += 1
                    return num_params, num_none, num_tensor
                n_total, n_none, n_tensor = check_grads_none(self.vlm)

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                if overwatch.is_rank_zero() and ((metrics.global_step + 1) % self.val_interval == 0):
                    self.run_validation(val_loader, action_tokenizer, metrics, max_val_batches=500)

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

    def unwrap_all(self, m):
        # return m.module if isinstance(m, FSDP) else m
        while isinstance(m, FSDP): m = m.module
        return m
    
    def locate_llm_params(self, vlm):
        base = self.unwrap_all(vlm)                       # PrismaticVLM
        llm_backbone = self.unwrap_all(base.llm_backbone) # your backbone wrapper
        llama = getattr(llm_backbone, "llm", llm_backbone)
        llama = self.unwrap_all(llama)                    # transformers LlamaForCausalLM or similar
        core = getattr(llama, "model", llama)        # handle models that keep core under .model

        embed = getattr(getattr(core, "embed_tokens", None), "weight", None)
        head  = getattr(getattr(llama, "lm_head", None), "weight", None)
        return embed, head, llama
    
    def show_llm(self, tag, vlm):
        # base = self.unwrap(vlm)
        # llm  = self.unwrap(base.llm_backbone)
        # try:
        #     emb = llm.get_input_embeddings().weight
        # except Exception:
        #     emb = getattr(getattr(llm, "model", None), "embed_tokens", None)
        #     emb = getattr(emb, "weight", None)
        # try:
        #     outm = llm.get_output_embeddings()
        #     head = getattr(outm, "weight", None)
        # except Exception:
        #     head = getattr(getattr(llm, "lm_head", None), "weight", None)

        # print(f"[{tag}] emb shape={None if emb is None else tuple(emb.shape)}  id={None if emb is None else id(emb)}")
        # print(f"[{tag}] head shape={None if head is None else tuple(head.shape)} id={None if head is None else id(head)}")
        emb, head, llama = self.locate_llm_params(vlm)
        print(f"[{tag}] emb:", None if emb is None else (tuple(emb.shape), id(emb), emb.data_ptr()))
        print(f"[{tag}] head:", None if head is None else (tuple(head.shape), id(head), head.data_ptr()))
        # Optional: are they actually tied?
        if emb is not None and head is not None:
            print(f"[{tag}] tied_storage:", emb.data_ptr() == head.data_ptr())

    def trace_shapes(self, mod):
        for name, p in mod.named_parameters():
            if p.ndim == 2 and p.shape == (32064, 4096):
                print("[TRACE-2D]", name, p.shape, p.data_ptr())
            elif p.ndim == 1 and p.numel() == 32064*4096:
                print("[TRACE-1D]", name, p.shape, p.data_ptr())

    def report_handles(self, mod):
        for name, m in mod.named_modules():
            if isinstance(m, FSDP):
                # List the parameters that this handle will manage
                try:
                    for pn, pp in m.named_parameters(recurse=False):
                        print(f"[FSDP-HANDLE] {name or '<ROOT>'} :: {pn} shape={pp.shape} data_ptr={pp.data_ptr()}")
                except Exception:
                    pass

    def move_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.move_to_device(v, device) for v in batch]
        else:
            return batch                    

    def run_validation(self, val_dataloader, action_tokenizer, metrics, 
                       max_val_batches: int | None = None):
        print("Start validation!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        print("[DEBUG] val_dataloader type:", type(val_dataloader))
        print("[DEBUG] val_dataloader.dataset type:", type(val_dataloader.dataset))
        print("[DEBUG] len(val_dataloader):", len(val_dataloader))

        try:
            it = iter(val_dataloader)
            first_batch = next(it)
            print("[DEBUG] Got first val batch! keys:", first_batch.keys())
            print("[DEBUG] input_ids shape:", first_batch["input_ids"].shape)
        except StopIteration:
            print("[DEBUG] val_dataloader is EMPTY (StopIteration).")
        except Exception as e:
            print("[DEBUG] Error while iterating val_dataloader:", repr(e))

        self.vlm.eval()

        total_loss = 0.0
        total_correct = 0
        total_mask = 0
        total_l1 = 0.0
        n_batches = 0

        status = metrics.get_status()
        with tqdm(
            total=(
                len(val_dataloader)
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            with torch.no_grad():
                for batch in val_dataloader:
                    n_batches += 1
                    if (max_val_batches is not None) and (n_batches > max_val_batches):
                        break

                    # Move batch to device
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    batch = self.move_to_device(batch, device)

                    with torch.autocast(
                        "cuda",
                        dtype=torch.float16,   # or self.mixed_precision_dtype
                        enabled=True,          # or self.enable_mixed_precision_training
                    ):
                        output = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            val=True,
                        )
                        loss = output.loss

                    # === Same action metrics as train, but no backward ===
                    action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = action_gt > action_tokenizer.action_token_begin_idx

                    correct = ((action_preds == action_gt) & mask).sum().float()
                    mask_count = mask.sum().float()

                    # Continuous actions
                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    l1 = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                    total_loss += loss.item()
                    total_correct += correct.item()
                    total_mask += mask_count.item()
                    total_l1 += l1.item()

                    # --- update tqdm in batches ---
                    progress.set_postfix_str(
                        f"{n_batches}/{len(val_dataloader)} batches"
                    )

            # Reduce across workers if distributed
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = torch.tensor([total_loss, total_correct, total_mask, total_l1, n_batches], device=device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                total_loss, total_correct, total_mask, total_l1, n_batches = tensor.tolist()

            avg_loss = total_loss / max(n_batches, 1)
            avg_acc = total_correct / max(total_mask, 1e-8)
            avg_l1 = total_l1 / max(n_batches, 1)

            # Log validation metrics (names up to you)
            metrics.commit_validation(
                loss=torch.tensor(avg_loss, device=device),
                l1_loss=torch.tensor(avg_l1, device=device),
                action_accuracy=torch.tensor(avg_acc, device=device),
            )
            val_status = metrics.push_validation()
            print(val_status)

            self.vlm.train()