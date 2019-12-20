import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigsoftmax(logits):
    max_values = torch.max(logits, -1, keepdim=True)[0]
    exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
    sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(-1, keepdim=True)

    return exp_logits_sigmoided / sum_exp_logits_sigmoided


def log_sigsoftmax(logits):
    max_values = torch.max(logits, -1, keepdim=True)[0]
    exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
    sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(-1, keepdim=True)
    log_probs = logits - max_values + torch.log(torch.sigmoid(logits)) - torch.log(sum_exp_logits_sigmoided)

    return log_probs


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, log_probs, target_probs):
        cross_entropy = -(target_probs * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        if self.reduction == 'sum':
            return cross_entropy.sum()
        return cross_entropy
        

class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            self.criterion = CrossEntropyLoss(reduction=reduction)
            n_ignore_idxs = 1 + (ignore_index >= 0)
            one_hot = torch.full((1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs)))
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        log_inputs = F.log_softmax(inputs, dim=-1)
        if self.confidence < 1:
            tdata = targets.data
            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp

        return self.criterion(log_inputs, targets)


class AdaCosLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss(reduction='mean')

    def forward(self, model, logits, targets):
        cosine = logits / model.scale

        if model.training:
            theta = torch.acos(cosine)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, targets.view(-1, 1), 1)
            with torch.no_grad():
                B_avg = torch.where(one_hot < 1, torch.exp(model.scale * cosine), torch.zeros_like(cosine))
                B_avg = torch.sum(B_avg) / cosine.shape[0]
                theta_med = torch.median(theta[one_hot == 1])
                model.scale = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))

        logits = cosine * model.scale

        loss = self.criterion(log_sigsoftmax(logits), targets)

        return loss


class FrozenDropout(nn.Module):
    @staticmethod
    def freeze_dropout(model, freeze_flag):
        def helper_func(module):
            for child_name, child in module.named_children():
                if isinstance(child, nn.Dropout):
                    setattr(module, child_name, FrozenDropout(p=child.p))

                if isinstance(child, FrozenDropout):
                    child.freeze(freeze_flag)
                else:
                    helper_func(child)

        helper_func(model)

    def __init__(self, p=0):
        super().__init__()

        if p < 0 or p >= 1:
            raise ValueError(f'Wrong parameter: expected 0 <= p < 1, got {p}')

        self.p = p
        self.freeze_flag = False
        self.cached_mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        if self.freeze_flag and self.cached_mask is not None:
            return x * self.cached_mask

        mask = torch.empty_like(x).bernoulli_(1 - self.p).mul_(1 - self.p)

        if self.freeze_flag:
            self.cached_mask = mask

        return x * mask

    def freeze(self, freeze_flag):
        if self.freeze_flag != freeze_flag:
            self.cached_mask = None
            self.freeze_flag = freeze_flag


class VATLoss(nn.Module):
    """
    VAT loss: https://arxiv.org/abs/1704.03976
    """

    @staticmethod
    def _l2_normalize(d, eps=1e-12):
        shape = d.shape
        d = d.view(shape[0], -1)
        d = F.normalize(d, p=2, dim=-1)
        d = d.view(*shape)

        return d

    def __init__(self, eps=1, n_iter=1):
        super().__init__()

        self.eps = eps
        self.n_iter = n_iter

    def forward(self, model, x):
        FrozenDropout.freeze_dropout(model, True)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
            mask = 1 - x.eq(model.padding_idx).unsqueeze(-1).float()

        d = torch.randn(*x.shape, model.embedding_dim, device=x.device)
        d = self._l2_normalize(d * mask)

        for _ in range(self.n_iter):
            d.requires_grad_()
            adv_logits = model(x, self.eps * d)
            adv_distance = F.binary_cross_entropy_with_logits(adv_logits, probs)
            grad = torch.autograd.grad(adv_distance, [d], retain_graph=False)[0]
            d = self._l2_normalize(grad.detach() * mask)

        adv_logits = model(x, self.eps * d)
        loss = F.binary_cross_entropy_with_logits(adv_logits, probs)

        FrozenDropout.freeze_dropout(model, False)

        return loss


def binary_entropy_with_logits(logits):
    probs = torch.sigmoid(logits)
    entropy = F.binary_cross_entropy_with_logits(logits, probs)
    
    return entropy