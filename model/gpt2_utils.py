import json
import urllib.request
import tempfile
import copy
import os

import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
from scipy.interpolate import RectBivariateSpline


MODEL_INFO = {
    'gpt2_small': {
        'base_url': 'https://storage.googleapis.com/gpt-2/models/117M/',
        'weights': ['checkpoint', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'],
        'bpe_vocab': 'encoder.json',
        'bpe_codes': 'vocab.bpe',
        'config': {
            'n_layers': 12,
            'n_embeddings': 50257,
            'n_pos_embeddings': 1024,
            'embeddings_size': 768,
            'n_heads': 12
        }
    },
    'gpt2_medium': {
        'base_url': 'https://storage.googleapis.com/gpt-2/models/345M/',
        'weights': ['checkpoint', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'],
        'bpe_vocab': 'encoder.json',
        'bpe_codes': 'vocab.bpe',
        'config': {
            'n_layers': 24,
            'n_embeddings': 50257,
            'n_pos_embeddings': 1024,
            'embeddings_size': 1024,
            'n_heads': 16
        }
    },
    'gpt2_large': {
        'base_url': 'https://storage.googleapis.com/gpt-2/models/774M/',
        'weights': ['checkpoint', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'],
        'bpe_vocab': 'encoder.json',
        'bpe_codes': 'vocab.bpe',
        'config': {
            'n_layers': 36,
            'n_embeddings': 50257,
            'n_pos_embeddings': 1024,
            'embeddings_size': 1280,
            'n_heads': 20,
        }
    }
}


def _download_file(file_url, output_path):
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(file_url, output_path)


def _get_gpt2_weights(params_dir, model):
    def get_weights(name, transpose=False):
        weights = tf.train.load_variable(params_dir, name)
        if transpose:
            weights = weights.squeeze(0).transpose((1, 0))
        return torch.from_numpy(weights)

    pos_embedding = get_weights('model/wpe')
    tok_embedding = get_weights('model/wte')

    n_layers = MODEL_INFO[model]['config']['n_layers']
    transformer_state = {'final_norm.weight': get_weights('model/ln_f/g'),
                         'final_norm.bias': get_weights('model/ln_f/b')}
    for layer_id in range(n_layers):
        layer_state = {f'layers.{layer_id}.attn.qkv_proj.weight': get_weights(f'model/h{layer_id}/attn/c_attn/w', transpose=True),
                       f'layers.{layer_id}.attn.qkv_proj.bias': get_weights(f'model/h{layer_id}/attn/c_attn/b'),
                       f'layers.{layer_id}.attn.out_proj.weight': get_weights(f'model/h{layer_id}/attn/c_proj/w', transpose=True),
                       f'layers.{layer_id}.attn.out_proj.bias': get_weights(f'model/h{layer_id}/attn/c_proj/b'),
                       f'layers.{layer_id}.attn_norm.weight': get_weights(f'model/h{layer_id}/ln_1/g'),
                       f'layers.{layer_id}.attn_norm.bias': get_weights(f'model/h{layer_id}/ln_1/b'),
                       f'layers.{layer_id}.ff.layer_1.weight': get_weights(f'model/h{layer_id}/mlp/c_fc/w', transpose=True),
                       f'layers.{layer_id}.ff.layer_1.bias': get_weights(f'model/h{layer_id}/mlp/c_fc/b'),
                       f'layers.{layer_id}.ff.layer_2.weight': get_weights(f'model/h{layer_id}/mlp/c_proj/w', transpose=True),
                       f'layers.{layer_id}.ff.layer_2.bias': get_weights(f'model/h{layer_id}/mlp/c_proj/b'),
                       f'layers.{layer_id}.ff_norm.weight': get_weights(f'model/h{layer_id}/ln_2/g'),
                       f'layers.{layer_id}.ff_norm.bias': get_weights(f'model/h{layer_id}/ln_2/b')}

        transformer_state.update(layer_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def _check_supported_models(model):
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')


def prepare_gpt2_weights(output_path, model):
    _check_supported_models(model)

    if os.path.exists(output_path):
        return

    with tempfile.TemporaryDirectory() as params_dir:
        for file in MODEL_INFO[model]['weights']:
            file_url = MODEL_INFO[model]['base_url'] + file
            file_path = os.path.join(params_dir, file)
            _download_file(file_url, file_path)

        if model == 'gpt2_small' or \
             model == 'gpt2_medium' or \
             model == 'gpt2_large':
            weights = _get_gpt2_weights(params_dir, model)
        else:
            assert False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(weights, output_path)


def prepare_bpe_vocab(output_path, model):
    _check_supported_models(model)

    if os.path.exists(output_path):
        return

    file_url = MODEL_INFO[model]['base_url'] + MODEL_INFO[model]['bpe_vocab']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _download_file(file_url, output_path)


def prepare_bpe_codes(output_path, model):
    _check_supported_models(model)

    if os.path.exists(output_path):
        return

    file_url = MODEL_INFO[model]['base_url'] + MODEL_INFO[model]['bpe_codes']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _download_file(file_url, output_path)


def load_gpt2_weights(gpt_model, state, n_special_tokens=0):
    if isinstance(gpt_model.embedding.pos_embedding, nn.Embedding):
        if gpt_model.embedding.pos_embedding.num_embeddings - 1 > state['pos_embedding'].shape[0]:
            xx = np.linspace(0, state['pos_embedding'].shape[0], gpt_model.embedding.pos_embedding.num_embeddings - 1)
            new_kernel = RectBivariateSpline(np.arange(state['pos_embedding'].shape[0]),
                                             np.arange(state['pos_embedding'].shape[1]),
                                             state['pos_embedding'])
            state['pos_embedding'] = new_kernel(xx, np.arange(state['pos_embedding'].shape[1]))

        state['pos_embedding'] = state['pos_embedding'][:gpt_model.embedding.pos_embedding.num_embeddings - 1]
        gpt_model.embedding.pos_embedding.weight.data[1:] = state['pos_embedding']

    state['tok_embedding'] = state['tok_embedding'][:gpt_model.embedding.tok_embedding.num_embeddings - n_special_tokens]
    gpt_model.embedding.tok_embedding.weight.data[:n_special_tokens] = state['tok_embedding'].mean(dim=0)
    gpt_model.embedding.tok_embedding.weight.data[n_special_tokens:] = state['tok_embedding']

    gpt_model.load_state_dict(state['transformer_state'], strict=False)
