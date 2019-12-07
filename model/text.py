import json
import regex as re
import random


class GPT2Tokenizer:
    @staticmethod
    def _bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """

        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]

        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
                
        cs = [chr(n) for n in cs]
        mapping = dict(zip(bs, cs))

        return mapping

    def __init__(self):
        self.byte_encoder = GPT2Tokenizer._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def tokenize(self, text):
        tokens = []
        tokens = [''.join([self.byte_encoder[b] for b in t.encode('utf-8')]) for t in re.findall(self.pat, text)]
        tokens = [tuple(token) for token in tokens]

        return tokens

    def detokenize(self, bpe_tokens):
        text = ''.join(bpe_tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')

        return text


class BPEVocab:
    @staticmethod
    def from_files(vocab_path, codes_path, tokenizer, special_tokens):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version:'):
                codes = codes[1:]

            codes = [tuple(c.split()) for c in codes if c]

        return BPEVocab(vocab, codes, tokenizer, special_tokens)

    @staticmethod
    def _get_pairs(sequence, dropout=0):
        if len(sequence) < 2:
            return set()
        return set(p for p in zip(sequence[:-1], sequence[1:])
                   if not dropout or random.random() > dropout)

    def __init__(self, vocab, codes, tokenizer, special_tokens):
        assert isinstance(special_tokens, dict)  # token_name + token

        filtered_special_tokens = [t for t in special_tokens if t not in vocab]
        special_token2id = {t: i for i, t in enumerate(filtered_special_tokens)}
        token2id = {t: i + len(filtered_special_tokens) for t, i in vocab.items()}
        token2id.update(special_token2id)

        self.special_tokens = special_tokens
        self.n_new_tokens = len(filtered_special_tokens)
        self.token2id = token2id
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.tokenizer = tokenizer
        self.cache = {}

        for token_name, token in special_tokens.items():
            setattr(self, token_name + '_id', self.token2id[token])

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return self.n_new_tokens

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.special_tokens]

    def _bpe(self, token, dropout=0):
        if token in self.cache and not dropout:
            return self.cache[token]

        word = token
        pairs = BPEVocab._get_pairs(word, dropout)

        if not pairs:
            return word

        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = BPEVocab._get_pairs(word, dropout)

        self.cache[token] = word

        return word

    def string2ids(self, string, dropout=0):
        tokens = self.tokenizer.tokenize(string)
        bpe_tokens = sum([self._bpe(t, dropout) for t in tokens], tuple())
        ids = [self.token2id[t] for t in bpe_tokens if t in self.token2id]

        return ids

    def ids2string(self, ids):
        bpe_tokens = [self.id2token[id] for id in ids]
        string = self.tokenizer.detokenize(bpe_tokens)

        return string
