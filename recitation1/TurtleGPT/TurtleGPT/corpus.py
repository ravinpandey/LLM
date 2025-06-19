import torch
from torch.utils.data import Dataset
from bpe import get_workspace, get_stats, apply_merge
import random

class Corpus(Dataset):

    def __init__(self, c_window_size, corpus_file, vocab_size=None,sft=False):
        super().__init__()
        data = open(corpus_file, 'r').read().replace('\n',' ')

        self.c_window_size = c_window_size
        chars = sorted(list(set(data + "#$")))

        self.ttoi = { ch:i for i,ch in enumerate(chars) }
        self.itot = { i:ch for i,ch in enumerate(chars) }
        self.sft = sft

        if vocab_size:
            bpe_train_sample = min(1000,len(data))
            i = len(chars)
            num_merges = vocab_size - i
            words = data[:bpe_train_sample].split(" ")

            workspace = get_workspace(words)
            for j in range(num_merges):
                pairs = get_stats(workspace)
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                workspace = apply_merge(best, workspace)
                self.ttoi[''.join(best)] = i
                self.itot[i] = ''.join(best)
                i += 1

        self.vocab_size = len(self.ttoi)
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data) - self.c_window_size

    def __getitem__(self, start):
        i=start
        iencoding = []

        try:
        
            for _ in range(self.c_window_size + 1):
                for cand in range(self.vocab_size,0,-1):
                    if self.data[i:].startswith(self.itot[cand-1]):
                        iencoding.append(cand-1)
                        i += len(self.itot[cand-1])
                        break

        except Exception as e:
            return self.__getitem__(random.randint(0,len(self.data)))

        x = torch.tensor(iencoding[:-1], dtype=torch.long)
        y = torch.tensor(iencoding[1:], dtype=torch.long)
        return x, y

    def encode(self,prompt):
        rv = []
        while prompt:
            for cand in range(self.vocab_size, 0, -1):
                if prompt.startswith(self.itot[cand - 1]):
                    rv.append(cand - 1)
                    prompt = prompt[len(self.itot[cand - 1]):]
                    break
        return rv

