from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from kymatio.torch import Scattering1D

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            in_channels: int = 3
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class ScatteringTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(self,
                 scattering_j: int = 6,
                 scattering_q: int = 8,
                 scattering_t: int = None,
                 layers: int = 2,
                 width: int = 768,
                 heads: int = 12,
                 mlp_ratio: float = 4.0,
                 ls_init_value: float = None,
                 dropout: float = 0.0,
                 global_average_pool: bool = False,
                 attentional_pool: bool = False,
                 n_queries: int = 256,
                 attn_pooler_heads: int = 8,
                 output_dim: int = 512,
                 output_tokens: bool = False,
                 act_layer: Callable = nn.GELU,
                 norm_layer: Callable = LayerNorm
                 ):
        super().__init__()

        # CHANGED:
        # use either scattering transform or CNN architecture before transformer

        shape = 2500
        scale = width ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(1, 1, width))
        self.pos_drop = nn.Dropout(p=dropout)

        if scattering_j >= 0:
            self.use_scattering = True
            # TODO T is an optional low-pass filter, could drop that parameter
            # T=2**13

            # Scattering transformation
            self.scattering = Scattering1D(J=scattering_j, shape=shape, Q=scattering_q, T=scattering_t)

            # TODO is there a function for the signal length dimension as well?
            # this is running it to look up the output size
            dummy_signal = torch.randn(shape)
            dummy_output = self.scattering(dummy_signal)
            self.scattering_signal_length = dummy_output.shape[1]
            self.scattering_output_size = dummy_output.shape[0]
            self.pos_embed = nn.Parameter(scale * torch.randn(1, 12 + 1, width))
            # this is the same as the function below
            # self.scattering_output_size = self.scattering.output_size()

            # self.cls_token = nn.Parameter(scale * torch.randn(1, 1, width))
            # self.pos_embed = nn.Parameter(scale * torch.randn(1, 12 + 1, width))
            # self.pos_drop = nn.Dropout(p=dropout)

            # TODO dropping zero-order coefficients and log-scaling should be config parameters
            # scattering_output_size - 1 takes into account dropping zero-order coefficients
            self.input_projection = nn.Linear(in_features=(
                    (self.scattering_output_size - 1) * self.scattering_signal_length),
                out_features=width)

        else:

            #breakpoint()
            self.use_scattering = False
            self.initial_filters = 16
            #self.conv1 = SimpleCNN(input_length=2500, hidden_size=1024, output_size=512)

            # Define reconstruction layers for each token/lead
            self.cnn_layers = nn.ModuleList([
                SimpleCNN(input_length=2500, hidden_size=1024, output_size=768)
                for _ in range(12)  # Assuming 12 leads in ECG
            ])


            # TODO compute dimension on the fly
            #self.pos_embed = nn.Parameter(scale * torch.randn(1, 81, width))
            self.pos_embed = nn.Parameter(scale * torch.randn(1, 13, width))
            #dummy_signal = torch.randn(shape)
            #dummy_output = self.conv1(dummy_signal)
            #self.scattering_signal_length = #dummy_output.shape[1]
            #self.scattering_output_size = #dummy_output.shape[0]

            self.input_projection = nn.Linear(in_features=512, out_features=width)

            # #breakpoint()
            # self.use_scattering = False
            # self.initial_filters = 16
            # self.conv1 = MultiConvolution2D(initial_filters=self.initial_filters)
            # # TODO compute dimension on the fly
            # self.pos_embed = nn.Parameter(scale * torch.randn(1, 81, width))
            # #dummy_signal = torch.randn(shape)
            # #dummy_output = self.conv1(dummy_signal)
            # #self.scattering_signal_length = #dummy_output.shape[1]
            # #self.scattering_output_size = #dummy_output.shape[0]
            #
            # self.input_projection = nn.Linear(in_features=int(self.initial_filters * 16 * 3), out_features=width)


        # TODO could also use a Conv2d layer here
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size,
        #                       bias=False)

        # Transformer network
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # Whether to return tokens as well
        self.output_tokens = output_tokens

    def _global_pool(self, x):
        return x.mean(dim=1), x

    def forward(self, x):

        batch_size, leads, _ = x.shape

        if self.use_scattering:
            #batch_size, leads, _ = x.shape
            # Reshape the input to treat each lead as a separate signal
            x_reshaped = x.view(-1, 2500)

            # Apply the scattering transform
            scattered_x = self.scattering.forward(x_reshaped)

            # log-scale and dropping zero-order coefficients
            scattered_x = torch.log(torch.abs(scattered_x[:, 1:, :]) + 1e-6)

            # Reshape to [64, 12, 125, 39]
            x1 = scattered_x.view(batch_size,
                                  leads,
                                  self.scattering_output_size - 1,
                                  self.scattering_signal_length)

            # Reshape to [64, 12, 125 * 39]
            x2 = x1.contiguous().view(batch_size,
                                      leads,
                                      (self.scattering_output_size - 1) * self.scattering_signal_length)

            # use this to project down to transformer input width
            x = self.input_projection(x2)

            # # concatenate the CLS token, and add the position tokens to each lead
            # x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
            # x = x + self.pos_embed
            # x = self.pos_drop(x)
            #
            # # TODO verify the dimensions here
            # x = x.permute(1, 0, 2)  # NLD -> LND
            # x = self.transformer(x)
            # x = x.permute(1, 0, 2)  # LND -> NLD

        else:

            # use CNN architecture
            batch_size, leads, _ = x.shape

            #from IPython import embed
            #embed()

            # # TODO is the multiconv2d implemenation using channels last?
            # x = x.unsqueeze(2)
            # x = x.permute(0, 2, 3, 1)
            # x = self.conv1.forward(x)

            #x2 = self.conv1.forward(x)
            # using same conv layer of all leads, but processing each lead separately
            #x2 = [self.conv1.forward(x_) for x_ in x.transpose(0,1)]

            x = [layer(lead.unsqueeze(1)) for layer, lead in zip(self.cnn_layers, x.transpose(0, 1))]
            x = torch.stack(x, dim=0)
            x = x.transpose(0, 1)

            #x_concat = torch.cat(x2, dim=2)

            #x2 = [layer(lead) for layer, lead in zip(self.cnn_layers, x)]

            # # [8, 768, 8, 10]
            # # this is batch, dim, height, width
            # # need [8*10, 8, 768] as input for transformer model
            #
            # x = x.permute(0, 2, 3, 1)  # permute to [batch, height, width, dim]
            # new_dim = x.size(1) * x.size(2)  # joint dim size
            #
            # # TODO do this in one step or clean up code
            # # Reshape the tensor to merge height and width
            # x = x.reshape(x.size(0), new_dim, x.size(3))  # Reshaping to [8, 8*10, 768]
            #x = self.input_projection(x)

        # concatenate the CLS token, and add the position tokens to each lead
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        return pooled


class PooledReconstructionWrapper(nn.Module):
    def __init__(self, visual_model, input_features=None, hidden_features=1024, output_features=None):
        super(PooledReconstructionWrapper, self).__init__()
        self.visual_model = visual_model

        if not input_features:
            self.reconstruction_layer = nn.Linear(512, 12 * 2500)
        else:
            self.reconstruction_layer1 = nn.Linear(input_features, hidden_features)
            self.reconstruction_layer2 = nn.Linear(hidden_features, hidden_features)
            self.reconstruction_output_layer = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        # Pass input through the visual model
        pooled, tokens = self.visual_model(x)

        x = F.relu(self.reconstruction_layer1(pooled))
        x = F.relu(self.reconstruction_layer2(x))
        reconstruction = self.reconstruction_output_layer(x)

        return pooled, tokens, reconstruction


class TokenReconstructionWrapper(nn.Module):
    def __init__(self, visual_model, token_features, hidden_features, output_features_per_lead):
        super(TokenReconstructionWrapper, self).__init__()
        self.visual_model = visual_model

        # Assuming token_features is the size of each token
        # hidden_features is the size of the hidden layer for each lead
        # output_features_per_lead is the number of features (data points) in each lead

        # Define reconstruction layers for each token/lead
        self.reconstruction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, output_features_per_lead)
            ) for _ in range(12)  # Assuming 12 leads in ECG
        ])

    def forward(self, x):
        # Pass input through the visual model
        pooled, tokens = self.visual_model(x)

        #from IPython import embed
        #embed()

        # TODO do they need to transposed?
        # Process each token through its respective reconstruction layer
        # remove the CLS token
        reconstructions = [layer(token) for layer, token in zip(self.reconstruction_layers, tokens[:,:-1,:].transpose(0, 1))]
        #reconstructions = [layer(token) for layer, token in zip(self.reconstruction_layers, tokens.transpose(1, 2))]
        #reconstructions = [layer(token) for layer, token in zip(self.reconstruction_layers, tokens)]

        # Stack the reconstructions to form the final output
        reconstruction = torch.stack(reconstructions, dim=1)

        return pooled, tokens, reconstruction


# class CPCScatteringTransformer(nn.Module):
#     def __init__(self, scattering_transformer, prediction_steps=12):
#         super().__init__()
#         self.scattering_transformer = scattering_transformer
#         self.prediction_steps = prediction_steps
#         self.cpc_head = nn.Linear(scattering_transformer.output_dim, scattering_transformer.output_dim)
#
#     def forward(self, x):
#         # Forward pass through the base transformer
#         base_representation = self.scattering_transformer(x)
#
#         # CPC prediction
#         cpc_prediction = self.cpc_head(base_representation)
#
#         return base_representation, cpc_prediction
#
#     @staticmethod
#     def info_nce_loss(current_rep, future_rep):
#         # Implement the InfoNCE loss function
#         # ...
#         return loss


class SimpleCNN(nn.Module):
    def __init__(self, input_length, hidden_size, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_length // 2), hidden_size)  # Adjust input_length
        self.fc2 = nn.Linear(hidden_size, output_size)  # output_size depends on the task

    def forward(self, x):
        x = self.pool1(self.bn1(self.act1(self.conv1(x))))
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiLayerCNN(nn.Module):
    def __init__(self, input_length, hidden_size, output_size):
        super(MultiLayerCNN, self).__init__()

        # Number of times to repeat the layers
        num_repeats = 4

        # Initial in_channels for the first Conv1d layer
        in_channels = 1
        out_channels = 32
        features = 16

        # Creating repeated layers
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, features, kernel_size=7, stride=2, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(features),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            # Update in_channels for the next sequence of layers
            in_channels = 32
            in_channels = 32
            in_channels = 32

        self.layers.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * (input_length // 2), hidden_size),  # Adjust input_length
                nn.Linear(hidden_size, output_size)  # output_size depends on the task
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class multi_conv2d_new(nn.Module):

    def __init__(self, in_filters, num_filters):
        super(multi_conv2d_new, self).__init__()
        self.a1 = nn.Conv2d(in_filters, int(num_filters / 1.5), kernel_size=(3, 3), padding='same')
        self.act_a1 = nn.ReLU()
        self.bn_a1 = nn.BatchNorm2d(int(num_filters / 1.5), affine=True)
        self.a2 = nn.Conv2d(int(num_filters / 1.5), num_filters, kernel_size=(1, 1), padding='same')
        self.act_a2 = nn.ReLU()
        self.bn_a2 = nn.BatchNorm2d(num_filters, affine=True)

        self.b1 = nn.Conv2d(in_filters, int(num_filters / 1.5), kernel_size=(7, 7), padding='same')
        self.act_b1 = nn.ReLU()
        self.bn_b1 = nn.BatchNorm2d(int(num_filters / 1.5), affine=True)
        self.b2 = nn.Conv2d(int(num_filters / 1.5), num_filters, kernel_size=(3, 3), padding='same')
        self.act_b2 = nn.ReLU()
        self.bn_b2 = nn.BatchNorm2d(num_filters, affine=True)
        self.b3 = nn.Conv2d(num_filters, num_filters, kernel_size=(1, 1), padding='same')
        self.act_b3 = nn.ReLU()
        self.bn_b3 = nn.BatchNorm2d(num_filters, affine=True)

        self.c1 = nn.Conv2d(in_filters, num_filters, kernel_size=(1, 1), padding='same')
        self.act_c1 = nn.ReLU()
        self.bn_c1 = nn.BatchNorm2d(num_filters, affine=True)

    #         self.res = nn.Sequential(nn.Conv2d(num_filters * 3 , num_filters * 3 * 2, kernel_size=3, stride=1, padding=1),
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(num_filters * 3 * 2, num_filters * 3, kernel_size=1, stride=1))

    def forward(self, x):
        a = self.act_a1(self.a1(x))
        a = self.bn_a1(a)
        a = self.act_a2(self.a2(a))
        a = self.bn_a2(a)

        b = self.act_b1(self.b1(x))
        b = self.bn_b1(b)
        b = self.act_b2(self.b2(b))
        b = self.bn_b2(b)
        b = self.act_b3(self.b3(b))
        b = self.bn_b3(b)

        c = self.act_c1(self.c1(x))
        c = self.bn_c1(c)

        #         m = x.mean(1).unsqueeze(1)
        #         out = torch.cat((m,a,b,c), dim=1)

        out = torch.cat((a, b, c), dim=1)

        #         out = out + self.res(out)
        #         out = out + x.mean(1).unsqueeze(1).repeat_interleave(out.shape[1],1)

        return out

class MultiConvolution2D(nn.Module):

    def __init__(self, initial_filters=64):
        super(MultiConvolution2D, self).__init__()

        self.conv1 = nn.Conv2d(1, initial_filters, kernel_size=(7, 3), stride=(2, 1))
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(initial_filters, affine=True)

        self.multi_conv2d_1 = multi_conv2d_new(initial_filters, initial_filters)
        self.multi_conv2d_2 = multi_conv2d_new(int(initial_filters * 3), initial_filters)
        self.mp1 = nn.MaxPool2d((3, 1))

        self.multi_conv2d_3 = multi_conv2d_new(int(initial_filters * 3), int(initial_filters * 1.5))
        self.multi_conv2d_4 = multi_conv2d_new(int(initial_filters * 1.5 * 3), int(initial_filters * 1.5))
        self.mp2 = nn.MaxPool2d((3, 1))

        self.multi_conv2d_5 = multi_conv2d_new(int(initial_filters * 1.5 * 3), int(initial_filters * 2))
        self.multi_conv2d_6 = multi_conv2d_new(int(initial_filters * 2 * 3), int(initial_filters * 2))
        self.mp3 = nn.MaxPool2d((2, 1))

        self.multi_conv2d_7 = multi_conv2d_new(int(initial_filters * 2 * 3), int(initial_filters * 3))
        self.multi_conv2d_8 = multi_conv2d_new(int(initial_filters * 3 * 3), int(initial_filters * 3))
        self.multi_conv2d_9 = multi_conv2d_new(int(initial_filters * 3 * 3), int(initial_filters * 4))
        self.mp4 = nn.MaxPool2d((2, 1))

        self.multi_conv2d_10 = multi_conv2d_new(int(initial_filters * 4 * 3), int(initial_filters * 5))
        self.multi_conv2d_11 = multi_conv2d_new(int(initial_filters * 5 * 3), int(initial_filters * 6))
        self.multi_conv2d_12 = multi_conv2d_new(int(initial_filters * 6 * 3), int(initial_filters * 7))
        self.mp5 = nn.MaxPool2d((2, 1))

        self.multi_conv2d_13 = multi_conv2d_new(int(initial_filters * 7 * 3), int(initial_filters * 8))
        self.multi_conv2d_14 = multi_conv2d_new(int(initial_filters * 8 * 3), int(initial_filters * 8))
        self.multi_conv2d_15 = multi_conv2d_new(int(initial_filters * 8 * 3), int(initial_filters * 8))
        self.mp6 = nn.MaxPool2d((2, 1))

        self.multi_conv2d_16 = multi_conv2d_new(int(initial_filters * 8 * 3), int(initial_filters * 12))
        self.multi_conv2d_17 = multi_conv2d_new(int(initial_filters * 12 * 3), int(initial_filters * 14))
        self.multi_conv2d_18 = multi_conv2d_new(int(initial_filters * 14 * 3), int(initial_filters * 16))

        # TODO parameterize
        #self.dp = nn.Dropout(0.1)
        #self.linear = nn.Linear(int(initial_filters * 16 * 3), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)

        x = self.multi_conv2d_1(x)
        x = self.multi_conv2d_2(x)
        x = self.mp1(x)

        #         x = x + (torch.randn(x.size()).to(x.get_device()) * 0.1 + 0)

        x = self.multi_conv2d_3(x)
        x = self.multi_conv2d_4(x)
        x = self.mp2(x)

        x = self.multi_conv2d_5(x)
        x = self.multi_conv2d_6(x)
        x = self.mp3(x)

        x = self.multi_conv2d_7(x)
        x = self.multi_conv2d_8(x)
        x = self.multi_conv2d_9(x)
        x = self.mp4(x)

        x = self.multi_conv2d_10(x)
        x = self.multi_conv2d_11(x)
        x = self.multi_conv2d_12(x)
        x = self.mp5(x)

        x = self.multi_conv2d_13(x)
        x = self.multi_conv2d_14(x)
        x = self.multi_conv2d_15(x)
        x = self.mp6(x)

        x = self.multi_conv2d_16(x)
        x = self.multi_conv2d_17(x)
        x = self.multi_conv2d_18(x)

        #x = torch.mean(x, [2, 3])

        #         x = x + (torch.randn(x.size()).to(x.get_device()) * 0.1 + 0)

        #x = self.dp(x)
        #x = self.linear(x)

        return x