import os

import torch
from apex import amp, optimizers
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from model.optim import SWA
from model.losses import LabelSmoothingLoss, AdaCosLoss, VATLoss, entropy_with_logits


class AvgMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, value):
        self._sum += value
        self._count += 1

    def __call__(self):
        if self._count:
            return self._sum / self._count
        return 0


def chunks(sequences, chunk_size):
    return [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]


class Trainer:
    def __init__(self, model, chunk_size, optimizer_params={}, loss_params={}, amp_params={},
                 tb_dir=None, device=None, n_jobs=0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        device = torch.device(device)
        smoothing = loss_params.get('smoothing', 0)
        lm_weight = loss_params.get('lm_weight', 0)
        cls_weight = loss_params.get('cls_weight', 1)
        vat_weight = loss_params.get('vat_weight', 0)
        lr = optimizer_params.get('lr', 1e-3)
        lr_decay = optimizer_params.get('lr_decay', 0)
        weight_decay = optimizer_params.get('weight_decay', 0)
        swa = optimizer_params.get('swa', False)
        swa_start = optimizer_params.get('swa_start', None)
        swa_freq = optimizer_params.get('swa_freq', None)
        swa_lr = optimizer_params.get('swa_lr', None)
        warmap = optimizer_params.get('warmap', 1000)
        opt_level = amp_params.get('opt_level', 'O0')
        loss_scale = amp_params.get('loss_scale', None)

        torch.cuda.set_device(device)

        self.model = model.to(device)
        self.criterion = AdaCosLoss()
        self.vat_criterion = VATLoss(eps=1, n_iter=1)
        # TODO: add additional ignore indexs
        self.lm_criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings,
                                               ignore_index=self.model.padding_idx,
                                               smoothing=smoothing).to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'norm']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = optimizers.FusedAdam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        if swa:
            self.optimizer = SWA(self.optimizer, swa_start, swa_freq, swa_lr)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level, loss_scale=loss_scale)

        def scheduler_func(iteration):
            if iteration <= warmap:
                return iteration / warmap
            return max(1 - lr_decay * iteration, 1e-9)

        self.swa = swa
        self.scheduler = LambdaLR(self.optimizer, scheduler_func)
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.lm_weight = lm_weight
        self.cls_weight = cls_weight
        self.vat_weight = vat_weight
        self.last_epoch = 0
        self.chunk_size = chunk_size
        self.device = device
        self.n_jobs = n_jobs

    def _train_epoch(self, train_dataloader):
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f'Train, epoch #{self.last_epoch}')
        self.model.train()

        cls_loss, lm_loss, vat_loss, entropy = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for chunks_list in tqdm_train_dataloader:
            self.optimizer.zero_grad()

            for tokens, labels in chunks_list:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                chunk_vat_loss = self.vat_criterion(self.model, tokens)

                cls_logits, lm_logits = self.model(tokens)

                chunk_cls_loss = self.criterion(self.model, cls_logits.reshape(-1, cls_logits.shape[-1]), labels.reshape(-1))
                chunk_lm_loss = self.lm_criterion(lm_logits[:, :-1].reshape(-1, lm_logits.shape[-1]), tokens[:, 1:].reshape(-1))
                chunk_entropy = entropy_with_logits(cls_logits)
                full_loss = (self.vat_weight * chunk_vat_loss +
                             self.cls_weight * chunk_cls_loss +
                             self.lm_weight * chunk_lm_loss) / len(chunks_list)

                with amp.scale_loss(full_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                cls_loss.update(chunk_cls_loss.mean().item())
                lm_loss.update(chunk_lm_loss.item())
                vat_loss.update(chunk_vat_loss.item())
                entropy.update(chunk_entropy.item())

            self.optimizer.step()
            self.scheduler.step()

            tqdm_train_dataloader.set_postfix({'cls_loss': cls_loss(),
                                               'lm_loss': lm_loss(),
                                               'vat_loss': vat_loss(),
                                               'entropy': entropy()})

        if self.swa:
            self.optimizer.swap_swa_sgd()

        self.writer.add_scalar('train/cls_loss', cls_loss(), global_step=self.last_epoch)
        self.writer.add_scalar('train/lm_loss', lm_loss(), global_step=self.last_epoch)
        self.writer.add_scalar('train/vat_loss', vat_loss(), global_step=self.last_epoch)
        self.writer.add_scalar('train/entropy', entropy(), global_step=self.last_epoch)

    @torch.no_grad()
    def _test_epoch(self, test_dataloader):
        tqdm_test_dataloader = tqdm(test_dataloader, desc=f'Test, epoch #{self.last_epoch}')
        self.model.eval()

        cls_loss, lm_loss, entropy = AvgMeter(), AvgMeter(), AvgMeter()
        for chunks_list in tqdm_test_dataloader:
            for tokens, labels in chunks_list:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                cls_logits, lm_logits = self.model(tokens)

                chunk_cls_loss = self.criterion(self.model, cls_logits.reshape(-1, cls_logits.shape[-1]), labels.reshape(-1))
                chunk_lm_loss = self.lm_criterion(lm_logits[:, :-1].reshape(-1, lm_logits.shape[-1]), tokens[:, 1:].reshape(-1))
                chunk_entropy = entropy_with_logits(cls_logits)

                cls_loss.update(chunk_cls_loss.item())
                lm_loss.update(chunk_lm_loss.item())
                entropy.update(chunk_entropy.item())

            tqdm_test_dataloader.set_postfix({'cls_loss': cls_loss(),
                                              'lm_loss': lm_loss(),
                                              'entropy': entropy()})

        self.writer.add_scalar('test/cls_loss', cls_loss(), global_step=self.last_epoch)
        self.writer.add_scalar('test/lm_loss', lm_loss(), global_step=self.last_epoch)
        self.writer.add_scalar('test/entropy', entropy(), global_step=self.last_epoch)

        result_metric = -cls_loss()

        return result_metric

    def _save_checkpoint(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def _collate_func(self, data):
        tokens, labels = zip(*data)
        tokens_list = [pad_sequence(c, batch_first=True, padding_value=self.model.padding_idx)
                       for c in chunks(tokens, self.chunk_size)]
        labels_list = [torch.stack(c, dim=0) for c in chunks(labels, self.chunk_size)]

        return list(zip(tokens_list, labels_list))

    def train(self, train_data, n_epochs, batch_size, test_data=None, test_batch_size=None,
              last_checkpoint_path=None, best_checkpoint_path=None):

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=self._collate_func, num_workers=self.n_jobs)

        if test_data is not None:
            if test_batch_size is None:
                test_batch_size = batch_size
            test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False,
                                         collate_fn=self._collate_func, num_workers=self.n_jobs)

        best_metric = float('-inf')
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None:
                torch.cuda.empty_cache()
                metric = self._test_epoch(test_dataloader)

                if best_checkpoint_path is not None:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1
