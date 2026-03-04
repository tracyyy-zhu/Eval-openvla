# vggt_intermediate.py
import torch
import torch.nn as nn
from typing import List, Union, Iterable

from vggt.models.vggt import VGGT


class VGGTFeaturizer(nn.Module):
    """
    Adapter that wraps VGGT and exposes a DINO-style get_intermediate_layers
    plus a .blocks attribute, so existing DINO/SigLIP code can reuse it.
    """
    def __init__(self, hf_id: str = "facebook/VGGT-1B", device: str = "cuda"):
        super().__init__()
        self.vggt = VGGT.from_pretrained(hf_id)
        # self.vggt.to(device)
        self.device = device
        self.patch_embed = self.vggt.aggregator.patch_embed

        # For compatibility with code that does len(self.dino_featurizer.blocks)
        # We pretend we have `depth` "blocks", matching the aggregator’s depth.
        depth = self.vggt.aggregator.depth
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])

        # Convenience: where patch tokens start (after camera + register tokens)
        self.patch_start_idx = self.vggt.aggregator.patch_start_idx

        base_dim = getattr(self.vggt, "embed_dim", 1024)
        self.embed_dim = 2 * base_dim

    @torch.no_grad()
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Iterable[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = True,
        norm: bool = False,
    ) -> List[torch.Tensor]:
        """
        Emulate timm VisionTransformer.get_intermediate_layers:

        - x: (B, 3, H, W)  (we treat this as a single-view sequence S=1)
        - returns a list of tensors, each roughly like (B, N_tokens, D)

        We ignore `reshape` and `norm` flags, but keep them in the signature
        for compatibility with existing calls.
        """
        device = self.device
        x = x.to(device)

        # VGGT expects [B, S, 3, H, W] or [S, 3, H, W]; treat our input as S=1
        if x.dim() == 4:            # (B, 3, H, W)
            x = x.unsqueeze(1)      # -> (B, 1, 3, H, W)

        agg_list, patch_start_idx = self.vggt.aggregator(x)
        # agg_list is length = depth, each (B, S, P, 2*C)
        patches = agg_list[-1] # (32, 1, 261, 2048)

        depth = len(agg_list) # =24

        # # Support n as int or iterable-of-indices
        # if isinstance(n, int):
        #     print("n:", n)
        #     idxs = list(range(n, depth))  # last n
        # else:
        #     idxs = sorted(list(n))

        # outs: List[torch.Tensor] = []
        # for i in idxs:
        #     feat = agg_list[i]  # (B, S, P, D)
        #     B, S, P, D = feat.shape

        #     # If you have S>1 you can pick a view; for now we assume S=1 and drop it.
        #     feat = feat[:, 0]   # -> (B, P, D)

        #     # Optionally drop special tokens and keep only patch tokens
        #     if not return_class_token and patch_start_idx is not None and patch_start_idx > 0:
        #         feat = feat[:, patch_start_idx:, :]  # (B, P_patches, D)

        #     outs.append(feat)
        if not return_class_token:
            patches = self._vggt_last_patches(patches, patch_start_idx, batch_size=x.shape[0])
        # patches: [B, 256, 2048]
        return patches

    def _vggt_last_patches(
            self,
            last: torch.Tensor,
            patch_start_idx: int,
            batch_size: int,
        ) -> torch.Tensor:
            """
            last: [B*S, P, D] or [B, S, P, D]
            returns: [B, 256, D]  (16x16 patches flattened)
            """
            if last.dim() == 3:
                # last is [B*S, P, D] → infer S
                BS, P, D = last.shape
                assert BS % batch_size == 0, f"Cannot infer S from {last.shape} and B={batch_size}"
                S = BS // batch_size
                last = last.view(batch_size, S, P, D)   # [B, S, P, D]
            elif last.dim() == 4:
                B, S, P, D = last.shape
                assert B == batch_size, f"Batch mismatch: expected {batch_size}, got {B}"
            else:
                raise ValueError(f"Unexpected last shape: {last.shape}")

            # Keep only patch tokens: drop camera+registers (front)
            patches = last[:, 0, patch_start_idx:, :]   # [B, 256, D] (since 261 - 5 = 256)

            return patches