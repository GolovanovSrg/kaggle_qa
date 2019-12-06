import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import Transformer, DistanceLayer


class ClassificationModel(nn.Module):
    def __init__(self, n_outputs, n_classes, cls_embedding_dim, n_centers, n_layers, n_embeddings,
                 n_pos_embeddings, embedding_dim, n_heads, padding_idx, dropout=0, future_mask=True,
                 constant_pos_embedding=False, adapters_mode=False):
        super().__init__()

        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.cls_embedding_dim = cls_embedding_dim
        self.n_centers = n_centers
        self.n_layers = n_layers
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.future_mask = future_mask
        self.constant_pos_embedding = constant_pos_embedding
        self.adapters_mode = adapters_mode

        # TODO: normalized embeddings
        self.encoder = Transformer(n_layers=n_layers,
                                   n_embeddings=n_embeddings,
                                   n_pos_embeddings=n_pos_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   n_heads=n_heads,
                                   dropout=dropout,
                                   embedding_dropout=dropout,
                                   attn_dropout=dropout,
                                   ff_dropout=dropout,
                                   constant_pos_embedding=constant_pos_embedding,
                                   future_mask=future_mask,
                                   adapters_mode=adapters_mode)

        # TODO: add more layers
        self.proj = nn.Linear(embedding_dim, n_outputs * n_classes)
        # TODO: don't share classes
        self.dist = DistanceLayer(embedding_dim, n_classes,
                                  middle_feature=cls_embedding_dim,
                                  n_centers=n_centers)

    def named_parameters(self, *args, **kwargs):
        named_params = super().named_parameters(*args, **kwargs)
        if not self.adapters_mode:
            return named_params

        adapter_params = []
        for name, param in named_params:
            if 'adapter' in name or \
               'norm' in name or \
               'proj' in name or \
               'dist' in name:
                adapter_params.append((name, param))
            else:
                param.requires_grad_(False)

        return adapter_params

    @property
    def scale(self):
        return self.dist.scale

    @scale.setter
    def scale(self, x):
        self.dist.scale = x

    def parameters(self, *args, **kwargs):
        return (param for _, param in self.named_parameters(*args, **kwargs))

    def predict_from_logits(self, logits):
        return torch.argmax(logits, dim=-1)

    def predict(self, x):
        logits, _ = self.forward(x)
        return self.predict_from_logits(logits)

    def forward(self, x, emb_noise=0):
        x, padding_mask = self.encoder(x, emb_noise)
        lengths = (~padding_mask).long().sum(dim=-1)
        lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        cls_x = x.gather(1, lengths-1).squeeze(1)
        cls_x = self.proj(cls_x).reshape(-1, self.n_outputs, self.n_classes)
        cls_output = self.dist(cls_x)
        lm_output = F.linear(x, self.encoder.embedding.tok_embedding.weight)

        return cls_output, lm_output
