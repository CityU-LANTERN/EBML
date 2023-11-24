import pickle
from collections import Counter
import argparse
import math
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

class TorchVocab(object):
    """
    :property freqs: collections.Counter, コーパス中の単語の出現頻度を保持するオブジェクト
    :property stoi: collections.defaultdict, string → id の対応を示す辞書
    :property itos: collections.defaultdict, id → string の対応を示す辞書
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param counter: collections.Counter, データ中に含まれる単語の頻度を計測するためのcounter
        :param max_size: int, vocabularyの最大のサイズ. Noneの場合は最大値なし. defaultはNone
        :param min_freq: int, vocabulary中の単語の最低出現頻度. この数以下の出現回数の単語はvocabularyに加えられない.
        :param specials: list of str, vocabularyにあらかじめ登録するtoken
        :param vectors: list of vectors, 事前学習済みのベクトル. ex)Vocab.load_vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # special tokensの出現頻度はvocabulary作成の際にカウントされない
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # まず頻度でソートし、次に文字順で並び替える
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # 出現頻度がmin_freq未満のものはvocabに加えない
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # dictのk,vをいれかえてstoiを作成する
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size,
                         min_freq=min_freq)

    # override用
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in texts:
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

#
# from build_vocab import WordVocab
# from dataset import Seq2seqDataset

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                   num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                   dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.trfm(embedded, embedded)  # (T,B,H)
        out = self.out(hidden)  # (T,B,V)
        out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out

    @staticmethod
    def split(sm):
        '''
        function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
        input: A SMILES
        output: A string with space between words
        '''
        arr = []
        i = 0
        while i < len(sm) - 1:
            if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                             'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
                arr.append(sm[i])
                i += 1
            elif sm[i] == '%':
                arr.append(sm[i:i + 3])
                i += 3
            elif sm[i] == 'C' and sm[i + 1] == 'l':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'C' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'C' and sm[i + 1] == 'u':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'N' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'N' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'R' and sm[i + 1] == 'b':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'R' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'X' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'L' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'l':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 's':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'g':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'u':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'M' and sm[i + 1] == 'g':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'M' and sm[i + 1] == 'n':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'T' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'Z' and sm[i + 1] == 'n':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 's' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 's' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 't' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'H' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '2':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '3':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '4':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '2':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '3':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '4':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'K' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'F' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            else:
                arr.append(sm[i])
                i += 1
        if i == len(sm) - 1:
            arr.append(sm[i])
        return ' '.join(arr)


def get_inputs(sm, vocab):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles, vocab):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm, vocab)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


if __name__ == '__main__':
    vocab = WordVocab.load_vocab('src/datasets/drug/vocab.pkl')
    feature_extractor = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    feature_extractor.load_state_dict(torch.load('src/ebml_models/modules/trfm_12_23000.pkl'))
    for p in feature_extractor.parameters():
        p.requires_grad = False
    print('Total parameters:', sum(p.numel() for p in feature_extractor.parameters()))
    feature_extractor.eval()

    # test_smiles = ["O=C(N[C@H]1Cc2ccccc2[C@@H]1NC(=O)[C@H]1CCNC1)c1cc2cc(F)ccc2[nH]1"]

    # DATASET ="sbap_general_ec50_size"
    # DATASET = "lbap_core_ic50_size"
    # DATASET = "lbap_general_ic50_size"
    # DATASET ="lbap_core_ki_size"
    DATASET ="lbap_core_ic50_assay"


    # Opening JSON file
    f = open('src/datasets/drug/{0}/{1}.json'.format(DATASET,DATASET))

    data = json.load(f)['split']
    train = data['train'] + data['iid_val']## a list of dictionary, each dictionary element is a molecule
    test_id = data['iid_test']
    test_ood = data['ood_val']  + data['ood_test']
    splits = {'test_ood':test_ood,'test_iid': test_id,'train':train}
    input={}
    label={}
    for k,split in splits.items():
        input[k] = []
        label[k] = []

        split_df = pd.DataFrame.from_dict(split)
        dfs = dict(tuple(split_df.groupby('domain_id')))
        too_small = 0
        constant = 0
        # for domain_id,domain_df in tqdm(dfs.items()):
        for domain_id,domain_df in tqdm(dfs.items()):
            smiles = domain_df['smiles'].values.tolist()
            y = domain_df['reg_label'].values.tolist()
            assert len(smiles)==len(y)
            if len(y)<5:
                # print(f"{domain_id} in {k} has less than 3 samples")
                too_small +=1
                continue
            if np.std(y)==0:
                # print(f"{domain_id} in {k} has constant targets : {y}")
                constant +=1
                continue
            print(len(y))
            continue
            # process
            x_split = [feature_extractor.split(sm) for sm in smiles]
            xid, xseg = get_array(x_split, vocab)
            x = feature_extractor.encode(torch.t(xid))
            input[k].append(x)
            label[k].append(y)

        print(f'using {len(input[k])} assays for {k}, {too_small} has less than 5, {constant} has constant y')

    # with open(f'src/datasets/drug/{DATASET}/reg_inputs.pickle', 'wb') as handle:
    #     pickle.dump(input, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(f'src/datasets/drug/{DATASET}/reg_labels.pickle', 'wb') as handle:
    #     pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    # print(smiles.keys())





