import random

import pandas as pd
import torch

from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import Dataset


SPECIAL_TOKENS = {'pad': '<pad>',
                  'eos': '<|endoftext|>',
                  'qt_bos': '<qt_bos>',
                  'qt_eos': '<qt_eos>',
                  'qb_bos': '<qb_bos>',
                  'qb_eos': '<qb_eos>',
                  'a_bos': '<a_bos>',
                  'a_eos': '<a_eos>'}


class GUESTDataset(Dataset):
    @staticmethod
    def feature_names():
        return ['question_title',
                'question_body',
                'answer']

    @staticmethod
    def target_names():
        return ['question_asker_intent_understanding',
                'question_body_critical',
                'question_conversational',
                'question_expect_short_answer',
                'question_fact_seeking',
                'question_has_commonly_accepted_answer',
                'question_interestingness_others',
                'question_interestingness_self',
                'question_multi_intent',
                'question_not_really_a_question',
                'question_opinion_seeking',
                'question_type_choice',
                'question_type_compare',
                'question_type_consequence',
                'question_type_definition',
                'question_type_entity',
                'question_type_instructions',
                'question_type_procedure',
                'question_type_reason_explanation',
                'question_type_spelling',
                'question_well_written',
                'answer_helpful',
                'answer_level_of_information',
                'answer_plausible',
                'answer_relevance',
                'answer_satisfaction',
                'answer_type_instructions',
                'answer_type_procedure',
                'answer_type_reason_explanation',
                'answer_well_written']

    def __init__(self, data, vocab, bpe_dropout=0, shuffle_parts=False, max_positions=1024):
        super().__init__()

        self._data = data
        self._vocab = vocab
        self._bpe_dropout = bpe_dropout
        self._shuffle_parts = shuffle_parts
        self._max_positions = max_positions

    def _make_feature(self, name, idx):
        text = self._data.iloc[idx][name]
        tokens = self._vocab.string2ids(text, self._bpe_dropout)

        if name == 'question_title':
            tokens = [self._vocab.qt_bos_id] + tokens + [self._vocab.qt_eos_id]
        elif name == 'question_body':
            tokens = [self._vocab.qb_bos_id] + tokens + [self._vocab.qb_eos_id]
        elif name == 'answer':
            tokens = [self._vocab.a_bos_id] + tokens + [self._vocab.a_eos_id]
        else:
            assert False

        return tokens

    def _make_target(self, name, idx):
        return self._data.iloc[idx][name]

    def _trim(self, title, question, answer):
        t_len = len(title)
        q_len = len(question)
        a_len = len(answer)

        if (t_len + q_len + a_len) > self._max_positions - 1:
            q_max_len = (self._max_positions - 1 - t_len) // 2
            a_max_len = self._max_positions - 1 - t_len - q_max_len

            if a_max_len > a_len:
                a_new_len = a_len 
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len
            
            question = question[:q_new_len-1] + [question[-1]]
            answer = answer[:a_new_len-1] + [answer[-1]]
        
        return title, question, answer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        title, question, answer = [self._make_feature(name, idx) for name in self.feature_names()]
        title, question, answer = self._trim(title, question, answer)

        features = [title, question, answer]
        if self._shuffle_parts:
            random.shuffle(features)

        tokens = sum(features, []) + [self._vocab.eos_id]
        targets = [self._make_target(name, idx) for name in self.target_names()]

        tokens = torch.tensor(tokens, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float)

        return tokens, targets


def read_data(data_path, test_size=0, seed=0):
    def feature_mapper(string):
        string = ' '.join(string.split(' '))
        string = '\n'.join(string.split('\n'))
        return string

    data = pd.read_csv(data_path).fillna(' ')
    data.loc[:, GUESTDataset.feature_names()] = data[GUESTDataset.feature_names()].applymap(feature_mapper)

    if test_size == 0:
        return data

    stratifier = IterativeStratification(n_splits=2,
                                         order=2,
                                         sample_distribution_per_fold=[test_size, 1.0 - test_size],
                                         random_state=seed)
    train_indexes, test_indexes = next(stratifier.split(data, data.loc[:, GUESTDataset.target_names()]))
    train_data = data.iloc[train_indexes, :].reset_index(drop=True)
    test_data = data.iloc[test_indexes, :].reset_index(drop=True)

    print(f'Train data size: {len(train_data)}\nTest data size: {len(test_data)}')

    return train_data, test_data
