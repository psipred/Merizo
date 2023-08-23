import torch
import torch.nn.functional as F
from torch import nn

from model.utils.utils import clean_domains, clean_singletons

# https://github.com/rstrudel/segmenter/blob/master/segm/model/decoder.py

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, bias=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (qkv[0], qkv[1], qkv[2])
        qk = q @ k.transpose(-2, -1)

        if bias is not None:
            qk = qk + bias

        attn = qk 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = nn.Identity()

    def forward(self, x, mask=None, return_attention=False, bias=None):
        y, attn = self.attn(self.norm1(x), mask, bias)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.class_norm = nn.LayerNorm(n_cls)

        # Background prediction layers
        self.bg_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            batch_first=True,
            num_layers=2,
            dropout=0,
            bidirectional=True,
        )

        self.bg_out = nn.Linear(d_model, 2)

        # Confidence Layers
        self.conf_gru = nn.GRU(
            input_size=n_cls,
            hidden_size=d_model,
            batch_first=True,
            num_layers=2,
            dropout=0,
            bidirectional=True,
        )

        self.conf_gru_all = nn.GRU(
            input_size=n_cls,
            hidden_size=d_model,
            batch_first=True,
            num_layers=2,
            dropout=0,
            bidirectional=True,
        )

        self.conf_out_all = nn.Linear(d_model, 1)
        self.conf_out = nn.Linear(d_model, 1)

    def forward(self, s, bias):

        # Initialise tensor for cls embeddings
        cls_emb = self.cls_emb.expand(s.size(0), -1, -1)

        # Concat to input at dim 1
        x = torch.cat((s, cls_emb), 1)  # , bg_emb), 1)

        # Alibi
        if bias is not None:
            pad = self.n_cls  # + 2
            bias = F.pad(bias, (0, pad, 0, pad), "constant", 0)

        # MHA blocks
        for blk in self.blocks:
            x = blk(x, bias)

        x = self.decoder_norm(x)

        # separate out original input patches and cls embeddings
        features = x[:, : -self.n_cls] @ self.proj_patch
        classes = x[:, -self.n_cls :] @ self.proj_classes

        # Some normalisation
        features = features / features.norm(dim=-1, keepdim=True)
        classes = classes / classes.norm(dim=-1, keepdim=True)

        # Dot product of (1,nres,emb) . (1,nclass,emb) = (1,nres,nclass)
        domain_masks = self.class_norm(features @ classes.transpose(1, 2))
        bg_masks = self.bg_out(self.bg_gru(features)[0])

        # Get per-domain confidence based on predictions
        pred_ids = domain_masks.argmax(dim=-1).flatten()

        # Cleaning before x bg
        pred_ids = clean_domains(pred_ids, 50)
        pred_ids = clean_singletons(pred_ids, 10)

        # Apply bg residue mask
        dom_ids = pred_ids * bg_masks.argmax(dim=-1).flatten()

        # Get unique domain indices
        unique_ids = dom_ids[dom_ids.nonzero()].unique()

        # Iterate over each pred index and pass through gru to predict confidence
        conf_res = torch.zeros_like(dom_ids).float()
        
        for i, d in enumerate(unique_ids):
            dom = domain_masks[:, dom_ids == d]

            dom_conf = (
                self.conf_out(self.conf_gru(dom)[1][-1:, :, :])
                .flatten()
                .clamp(min=0, max=1)
            )

            conf_res[dom_ids == d] = dom_conf

        return dom_ids.reshape(-1), conf_res.reshape(-1)
    
