import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import CheckpointFunction

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm


class ConstantPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, init_n_embeddings=1024):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = 0
        self.register_buffer('_embedding', ConstantPositionalEmbedding.get_embedding(init_n_embeddings, self.embedding_dim))

    @staticmethod
    def get_embedding(n_embeddings, embedding_dim, device=None):
        n_embeddings += 1  # 0 is the padding

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
        emb = torch.arange(n_embeddings, dtype=torch.float, device=device).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(n_embeddings, -1)
        emb[0, :] = 0

        if embedding_dim % 2:
            emb = torch.cat([emb, torch.zeros(n_embeddings, 1, dtype=torch.float, device=device)], dim=1)

        return emb

    def forward(self, positions):
        batch_size, seq_length = positions.shape

        if seq_length >= self._embedding.shape[0]:
            self._embedding = ConstantPositionalEmbedding.get_embedding(seq_length,
                                                                        self.embedding_dim,
                                                                        self._embedding.device)

        positions = positions.view(-1)
        pos_embeddings = self._embedding.index_select(0, positions)
        pos_embeddings = pos_embeddings.view(batch_size, seq_length, -1)

        return pos_embeddings


class LearnablePositionalEmbedding(nn.Embedding):
    def __init__(self, embedding_dim, n_embeddings, sparse=False):
        n_embeddings += 1  # 0 is the padding
        super().__init__(n_embeddings, embedding_dim, padding_idx=0, sparse=sparse)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.01)


class CombinedEmbedding(nn.Module):
    def __init__(self, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx=None,
                 constant_pos_embedding=False, sparse=False):
        super().__init__()

        self.tok_padding_idx = padding_idx
        self.pos_padding_idx = 0

        self.tok_embedding = nn.Embedding(n_embeddings, embedding_dim,
                                          padding_idx=self.tok_padding_idx, sparse=sparse)
        if constant_pos_embedding:
            self.pos_embedding = ConstantPositionalEmbedding(embedding_dim, n_pos_embeddings)
        else:
            self.pos_embedding = LearnablePositionalEmbedding(embedding_dim, n_pos_embeddings, sparse=sparse)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embedding.weight, std=0.02)

    def forward(self, x):
        padding_mask = x[:, :].eq(self.tok_padding_idx)
        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_padding_idx)

        x = self.tok_embedding(x) + self.pos_embedding(positions)

        return x, padding_mask


class MultiheadAttention(nn.Module):
    @classmethod
    @torch.no_grad()
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1).bool()
        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout, future_mask=True, efficient_attn=False):
        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.future_mask = future_mask
        self.efficient_attn = efficient_attn
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _filtered_softmax(self, logits, top_p=0.95):
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        with torch.no_grad():
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask_to_remove = cumulative_probabilities > top_p
            mask_to_remove[..., 1:] = mask_to_remove[..., :-1].clone()
            mask_to_remove[..., 0] = 0
        sorted_logits.masked_fill_(mask_to_remove, float('-inf'))
        logits.scatter_(-1, sorted_indices, sorted_logits)

        return F.softmax(logits, dim=-1)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _efficient_attn(self, q, k, v, apply_future_mask, padding_mask):
        if apply_future_mask:
            raise ValueError('Efficient attention does not support future mask')

        if padding_mask is not None:
            k.masked_fill_(padding_mask.bool().unsqueeze(1).unsqueeze(2), float('-inf'))

        k = F.softmax(k, dim=-1)
        k = self.dropout(k)
        g = torch.matmul(k, v)

        q = F.softmax(q, dim=-1)
        q = self.dropout(q)
        out = torch.matmul(q, g)

        return out

    def _attn(self, q, k, v, apply_future_mask, padding_mask):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.bool().unsqueeze(1).unsqueeze(2), float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = self.future_mask
        elif kv_same:
            q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
            kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]

            query = F.linear(query, q_w, q_b)
            key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        if self.efficient_attn:
            x = self._efficient_attn(query, key, value, apply_future_mask, padding_mask)
        else:
            x = self._attn(query, key, value, apply_future_mask, padding_mask)

        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


def gelu(x):
    """
    GELU activation: https://arxiv.org/abs/1606.08415
    """
    sqrt_two = 1.4142135623730951
    cdf = (x / sqrt_two).erf_().add_(1.0).mul_(0.5)
    return x.mul_(cdf)


class FeedForward(nn.Module):
    def __init__(self, in_features, middle_features, dropout):
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = self.layer_1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class Adapter(nn.Module):
    """
    https://arxiv.org/pdf/1902.00751.pdf
    """

    def __init__(self, in_features, middle_features):
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.adapter_norm = LayerNorm(in_features)

        self._init_weights()

    def _init_weights(self) :
        nn.init.normal_(self.layer_1.weight, std=1e-3)
        self.layer_1.bias.data.zero_()
        nn.init.normal_(self.layer_2.weight, std=1e-3)
        self.layer_2.bias.data.zero_()

    def forward(self, x):
        residual = x

        x = self.adapter_norm(x)
        x = self.layer_1(x)
        x = gelu(x)
        x = self.layer_2(x)

        x = residual + x

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout=0, attn_dropout=0, ff_dropout=0,
                 future_mask=True, adapters_mode=False, attn_checkpoint=True):
        super().__init__()

        if attn_checkpoint:
            assert attn_dropout == 0

        self.attn_checkpoint = attn_checkpoint
        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout, future_mask)
        self.attn_norm = LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = LayerNorm(n_features)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        if adapters_mode:
            self.attn_adapter = Adapter(n_features, n_features)
            self.ff_adapter = Adapter(n_features, n_features)
        else:
            self.attn_adapter = None
            self.ff_adapter = None

    def _process_attn(self, x, padding_mask, contexts):
        # TODO: don't share attention, try not sum
    
        if contexts is None:
            contexts = []

        residual = x

        x = self.attn_norm(x)
        ext_contexts = [(x, padding_mask)] + contexts
        if self.attn_adapter is not None:
            with torch.no_grad():
                attns = (self.attn(x, c, c, m) for c, m in ext_contexts)
                x = sum(attns, 0) / len(ext_contexts)
                x = self.attn_dropout(x)
            x = self.attn_adapter(x)
        else:
            if self.training and self.attn_checkpoint:
                attns = (CheckpointFunction.apply(self.attn, 4, x, c, c, m) for c, m in ext_contexts)
            else:
                attns = (self.attn(x, c, c, m) for c, m in ext_contexts)
            x = sum(attns, 0) / len(ext_contexts)
            x = self.attn_dropout(x)

        x = residual + x

        return x

    def _process_ff(self, x):
        residual = x

        x = self.ff_norm(x)
        if self.ff_adapter is not None:
            with torch.no_grad():
                x = self.ff(x)
                x = self.ff_dropout(x)
            x = self.ff_adapter(x)
        else:
            x = self.ff(x)
            x = self.ff_dropout(x)

        x = residual + x

        return x

    def forward(self, x, padding_mask, contexts=None):
        '''
        contexts = [(context1, padding_mask1), ...]
        '''
        x = self._process_attn(x, padding_mask, contexts)
        x = self._process_ff(x)

        return x, padding_mask


class Transformer(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx, n_heads,
                 dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0, future_mask=True,
                 constant_pos_embedding=False, sparse_embedding=False, adapters_mode=False):
        super().__init__()

        self.embedding = CombinedEmbedding(n_embeddings=n_embeddings,
                                           n_pos_embeddings=n_pos_embeddings,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_idx,
                                           constant_pos_embedding=constant_pos_embedding,
                                           sparse=sparse_embedding)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        base_block = TransformerBlock(n_features=embedding_dim,
                                      n_heads=n_heads,
                                      dropout=dropout,
                                      attn_dropout=attn_dropout,
                                      ff_dropout=ff_dropout,
                                      future_mask=future_mask,
                                      adapters_mode=adapters_mode)
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_layers)])
        self.final_norm = LayerNorm(embedding_dim)

    def forward(self, x, emb_noise=None, contexts=None):
        x, padding_mask = self.embedding(x)
        if emb_noise is not None:
            x = x + emb_noise
        x = self.embedding_dropout(x)

        float_padding_mask = padding_mask.type_as(x).requires_grad_()
        for layer in self.layers:
            x, _ = layer(x, float_padding_mask, contexts=contexts)
        x = self.final_norm(x)

        return x, padding_mask


class DistanceLayer(nn.Module):
    def __init__(self, in_features, out_features, middle_feature=None, n_centers=1):
        super().__init__()

        if middle_feature is None:
            middle_feature = in_features

        self.proj = nn.Sequential(nn.Linear(in_features, middle_feature),
                                  nn.CELU(inplace=True),
                                  nn.Linear(middle_feature, middle_feature))
        self.clusters = nn.Parameter(torch.Tensor(n_centers * out_features, middle_feature))
        self.pooling = nn.MaxPool1d(kernel_size=n_centers, stride=n_centers)
        self.register_buffer('scale', torch.tensor(math.sqrt(2) * math.log(out_features - 1)))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj[0].weight, std=0.02)
        nn.init.normal_(self.proj[2].weight, std=0.02)
        nn.init.normal_(self.clusters)

    def forward(self, x):
        x = self.proj(x)
        x = F.linear(F.normalize(x, dim=-1), F.normalize(self.clusters, dim=-1))
        x = self.scale * x

        return x
