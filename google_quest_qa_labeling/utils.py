import os
import random

import torch

from model.text import GPT2Tokenizer, BPEVocab
from model.gpt2_utils import MODEL_INFO, prepare_gpt2_weights, prepare_bpe_vocab, prepare_bpe_codes, load_gpt2_weights
from cls_model import ClassificationModel
from trainer import Trainer
from dataset import SPECIAL_TOKENS, GUESTDataset, read_data


def set_seed(seed=0):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def get_vocab(config):
    model_type = config['model_type']
    vocab_dir = config['vocab_dir']

    vocab_path = os.path.join(vocab_dir, '_gpt2_bpe.vocab')
    codes_path = os.path.join(vocab_dir, '_gpt2_bpe.codes')

    prepare_bpe_vocab(vocab_path, model_type)
    prepare_bpe_codes(codes_path, model_type)

    tokenizer = GPT2Tokenizer()

    return BPEVocab.from_files(vocab_path, codes_path, tokenizer, SPECIAL_TOKENS)


def get_model(config, vocab):
    model_type = config['model_type']
    parameters_dir = config['parameters_dir']
    checkpoint_path = config['restore_checkpoint_path']
    model_config = MODEL_INFO[model_type]['config']
    if config['n_pos_embeddings'] is None:
        config['n_pos_embeddings'] = model_config['n_pos_embeddings']

    model = ClassificationModel(n_outputs=config['n_outputs'],
                                n_classes=config['n_classes'],
                                cls_embedding_dim=config['cls_embedding_dim'],
                                n_centers=config['n_centers'],
                                n_layers=model_config['n_layers'],
                                n_embeddings=len(vocab),
                                n_pos_embeddings=config['n_pos_embeddings'],
                                embedding_dim=model_config['embeddings_size'],
                                n_heads=model_config['n_heads'],
                                padding_idx=vocab.pad_id,
                                dropout=config['dropout'],
                                future_mask=config['future_mask'],
                                constant_pos_embedding=config['constant_pos_embedding'],
                                adapters_mode=config['adapters_mode'])

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f'Checkpoint from {checkpoint_path}')
    elif parameters_dir is not None:
        parameters_path = os.path.join(parameters_dir, model_type + '_parameters.pt')
        prepare_gpt2_weights(parameters_path, model_type)
        parameters = torch.load(parameters_path, map_location='cpu')
        load_gpt2_weights(model.encoder, parameters, vocab.n_special_tokens)

    return model


def get_trainer(config, model):
    optimizer_params = {'lr': config['lr'],
                        'lr_decay': config['lr_decay'],
                        'weight_decay': config['weight_decay'],
                        'warmup': config['warmup'],
                        'swa': config['swa'],
                        'swa_start': config['swa_start'],
                        'swa_freq': config['swa_freq'],
                        'swa_lr': config['swa_lr']}
    loss_params = {'smoothing': config['smoothing'],
                   'lm_weight': config['lm_weight'],
                   'cls_weight': config['cls_weight'],
                   'vat_weight': config['vat_weight']}
    amp_params = {'opt_level': config['opt_level'],
                  'loss_scale': config['loss_scale']}
    chunk_size = config['chunk_size']
    device = config['device']
    n_jobs = config['n_jobs']

    trainer = Trainer(model=model,
                      chunk_size=chunk_size,
                      optimizer_params=optimizer_params,
                      loss_params=loss_params,
                      amp_params=amp_params,
                      device=device,
                      n_jobs=n_jobs)

    return trainer


def get_train_val_datasets(config, vocab, max_positions):
    train_data, val_data = read_data(data_path=config['train_data_path'],
                                     test_size=config['val_size'],
                                     seed=config['seed'])
    train_dataset = GUESTDataset(data=train_data,
                                 vocab=vocab,
                                 bpe_dropout=config['bpe_dropout'],
                                 shuffle_parts=config['shuffle_parts'],
                                 max_positions=max_positions)
    val_dataset = GUESTDataset(data=val_data,
                               vocab=vocab,
                               bpe_dropout=config['bpe_dropout'],
                               shuffle_parts=False,
                               max_positions=max_positions)

    return train_dataset, val_dataset
