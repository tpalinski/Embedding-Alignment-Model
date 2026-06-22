import torch
from torch import nn
import torch.nn.functional as F

from .AlignmentModelConfig import AlignmentModelConfig

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, audio, text):
        # text is detached (important ConST-style choice)
        attn_out, _ = self.attn(
            query=audio,
            key=text.detach(),
            value=text.detach(),
            need_weights=False,
        )

        return self.norm(audio + attn_out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)

class SharedProjection(nn.Module):
    def __init__(self, dim, emb_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x):
        return self.net(x)

class AlignmentModel(nn.Module):
    def __init__(
        self,
        config: AlignmentModelConfig,
        audio_adapter: nn.Module|None,
        text_adapter: nn.Module|None,
    ):
        super().__init__()

        self.config = config
        dim = config['input_dim']
        emb_dim = config['embedding_dim']
        num_heads = config['num_heads']

        self.audio_adapter = audio_adapter
        self.text_adapter = text_adapter

        self.cross_attn = CrossAttention(dim, num_heads)
        self.audio_sa = SelfAttentionBlock(dim, num_heads)
        self.text_sa = SelfAttentionBlock(dim, num_heads)

        self.pool = AttentionPool(dim)
        self.proj = SharedProjection(dim, emb_dim)


    def encode_audio(self, audio_tokens, text_tokens=None):
        if self.audio_adapter is not None:
            x = self.audio_adapter(audio_tokens)
        else:
            x = audio_tokens

        if self.training and text_tokens is not None:
            if self.text_adapter is not None:
                x = self.cross_attn(x, self.text_adapter(text_tokens))
            else:
                x = self.cross_attn(x, text_tokens)

        x = self.audio_sa(x)

        x = self.pool(x)
        x = self.proj(x)

        return F.normalize(x, dim=-1)

    def encode_text(self, text_tokens):
        if self.text_adapter is not None:
            x = self.text_adapter(text_tokens)
        else:
            x = text_tokens

        x = self.text_sa(x)

        x = self.pool(x)
        x = self.proj(x)

        return F.normalize(x, dim=-1)

    def forward(self, audio_tokens, text_tokens):
        if self.text_adapter is not None:
            x_t = self.text_adapter(text_tokens)
        else:
            x_t = text_tokens
        if self.audio_adapter is not None:
            x_a = self.audio_adapter(audio_tokens)
        else:
            x_a = audio_tokens
        x_a_guided = None
        if self.config['use_training_cross_attention'] and self.training and text_tokens is not None:
            x_a_guided = self.cross_attn(x_a, x_t)

        x_a_base = self.audio_sa(x_a)
        x_t_base = self.text_sa(x_t)

        a_base = self.proj(self.pool(x_a_base))
        t_base = self.proj(self.pool(x_t_base))

        a_base = F.normalize(a_base, dim=-1)
        t_base = F.normalize(t_base, dim=-1)

        outputs = {
            "a": a_base,
            "t": t_base,
        }

        # optional guided branch loss signal
        if x_a_guided is not None:
            x_a_guided = self.audio_sa(x_a_guided)
            a_guided = self.proj(self.pool(x_a_guided))
            outputs["a_guided"] = F.normalize(a_guided, dim=-1)

        return outputs
