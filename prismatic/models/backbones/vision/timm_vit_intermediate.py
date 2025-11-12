# timm_vit_intermediate.py
import torch
from torch import Tensor, nn
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

class TimmViTIntermediate(nn.Module):
    """
    Wraps a timm VisionTransformer (incl. EVA/EVA02 variants) and exposes:
      - get_intermediate_layers(x, n=1, return_cls=True, apply_norm=True)
    """
    def __init__(self, 
                 vit: nn.Module,
                 embed_dim: int = 1024,
                 n_storage_tokens: int = 0,
                 device: Any | None = None,):
        super().__init__()
        self.vit = vit
        self.patch_embed = vit.patch_embed
        self.cls_token   = getattr(vit, "cls_token", None)
        self.pos_embed   = getattr(vit, "pos_embed", None)
        self.pos_drop    = getattr(vit, "pos_drop", nn.Identity())
        self.blocks      = vit.blocks
        self.norm        = getattr(vit, "norm", None)
        self.patch_embed.flatten = True
        self.patch_embed.output_fmt = "NLC"
        self.embed_dim = getattr(vit, "embed_dim", None) or getattr(vit, "num_features", None)
        assert self.embed_dim is not None, "Could not infer ViT width (embed_dim/num_features)."
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.rope_embed = getattr(vit, "rope_embed", None)
        self.untie_cls_and_patch_norms = getattr(vit, "untie_cls_and_patch_norms", False)
        self.cls_norm   = getattr(vit, "fc_norm", None)  # cls-only norm if untied
        self.has_cls_token = hasattr(vit, "cls_token")

    @torch.no_grad()
    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        # before [32, 3, 224, 224]
        x = self.patch_embed(x)               # [32, 196, 1024]
        B, _, _ = x.shape
        H, W = 14, 14

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    @torch.no_grad()
    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if getattr(self, "rope_embed", None) is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    @torch.no_grad()
    def get_intermediate_layers(
        self, x: torch.Tensor, *, 
        n: Union[int, Sequence] = 1, 
        reshape: bool = False, 
        return_cls: bool = False, 
        return_extra_tokens: bool = False,
        apply_norm: bool = True
    ):
        """
        Returns a list of `n` layer outputs from the last to earlier, like Meta's DINO:
          - if return_cls=True: each element is [B, D] (CLS token)
          - else: each element is [B, P, D] (patch tokens)
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if apply_norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_cls and not return_extra_tokens:
            return tuple(outputs)
        elif return_cls and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_cls and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_cls and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    # Optional: define a forward that just proxies to get_intermediate_layers
    def forward(self, x, n=1, return_cls=True, apply_norm=True):
        return self.get_intermediate_layers(x, n=n, return_cls=return_cls, apply_norm=apply_norm)
