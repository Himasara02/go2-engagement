#!/usr/bin/env python3
"""GRU sequence baseline (intention-to-interact style) for full_extra_baselines.py. Needs PyTorch:

    ~/miniconda3/envs/bench312/bin/python -P full_seq_train.py [folder]     (folder = <EVAL_OUT>_extra)

Model and training fixed before seeing any result: per-frame features (distance, approach, facing, gaze and the
three sigmas) standardised on the training frames; one-layer GRU (32 units) over the episode's frames in time order,
a linear layer + sigmoid per frame; loss = class-balanced binary cross-entropy of every frame against the episode
label; Adam, lr 3e-3, 15 epochs, batch 64 episodes (of similar length), seed 0, no early stopping, no validation on test recordings.
Standing: leave-one-recording-out (one model per recording, scored on every person-frame -> loo[(n_takes, n)]).
Walking (seq_train.npz present): one model on all standing episodes -> train (standing frames), test (walking).
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "full_extra")
torch.set_num_threads(8)


class GRU(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.rnn = nn.GRU(k, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return self.out(self.rnn(x)[0])[..., 0]


def seqs(F, I, mu, sd):
    """Episode tensors: valid frames only, in time order. Returns padded x, mask, row index."""
    Fz = (F - mu) / sd
    rows = [r[r >= 0] for r in I]
    L = max(1, max(len(r) for r in rows))
    x = np.zeros((len(rows), L, F.shape[1]), np.float32)
    m = np.zeros((len(rows), L), bool)
    ix = np.full((len(rows), L), -1, np.int64)
    for j, r in enumerate(rows):
        x[j, :len(r)] = Fz[r]
        m[j, :len(r)] = True
        ix[j, :len(r)] = r
    return torch.tensor(x), torch.tensor(m), ix


def fit(F, I, lab, sel):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rows = np.concatenate([r[r >= 0] for r in I[sel]])
    mu, sd = F[rows].mean(0), F[rows].std(0) + 1e-6
    x, m, _ = seqs(F, I[sel], mu, sd)
    y = torch.tensor(lab[sel].astype(np.float32))[:, None].expand_as(m)
    pos = float(lab[sel].mean())
    w = torch.where(y > 0.5, torch.tensor(0.5 / pos), torch.tensor(0.5 / (1 - pos))) * m
    net = GRU(F.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    lens = m.sum(1).numpy()
    for _ in range(15):
        # random batches of 64 episodes of similar length (shuffle, sort within blocks of 20 batches, shuffle
        # batches); each batch is cut to its longest episode. The GRU runs forward in time, so padding after
        # an episode's last frame never changes its outputs: this only saves time.
        order = rng.permutation(len(x))
        blocks = [order[b:b + 1280] for b in range(0, len(x), 1280)]
        batches = [blk[np.argsort(lens[blk], kind="stable")][s:s + 64] for blk in blocks for s in range(0, len(blk), 64)]
        for k in rng.permutation(len(batches)):
            j = torch.tensor(batches[k])
            T = int(lens[batches[k]].max())
            loss = (lossf(net(x[j, :T]), y[j, :T]) * w[j, :T]).sum() / w[j, :T].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, mu, sd


def score(net, mu, sd, F, I):
    x, m, ix = seqs(F, I, mu, sd)
    with torch.no_grad():
        p = torch.sigmoid(net(x)).numpy()
    out = np.zeros(len(F), np.float32)
    out[ix[m.numpy()]] = p[m.numpy()]
    return out


def main():
    te = np.load(os.path.join(DIR, "seq_test.npz"))
    if os.path.exists(os.path.join(DIR, "seq_train.npz")):
        tr = np.load(os.path.join(DIR, "seq_train.npz"))
        net, mu, sd = fit(tr["F"], tr["I"], tr["lab"], np.ones(len(tr["lab"]), bool))
        np.savez(os.path.join(DIR, "seq_probs.npz"), train=score(net, mu, sd, tr["F"], tr["I"]),
                 test=score(net, mu, sd, te["F"], te["I"]))
        print("walking: trained on", len(tr["lab"]), "standing episodes")
        return
    n_t = int(te["take_of"].max()) + 1
    with Pool(4) as pool:                                   # 4 folds at a time, 4 threads each
        loo = np.stack(pool.map(fold, range(n_t), chunksize=1))
    np.savez(os.path.join(DIR, "seq_probs.npz"), loo=loo)


def fold(t):
    torch.set_num_threads(4)
    te = np.load(os.path.join(DIR, "seq_test.npz"))
    F, I, lab, tk = te["F"], te["I"], te["lab"], te["take_of"]
    net, mu, sd = fit(F, I, lab, tk != t)
    print(f"fold {t + 1} done", flush=True)
    return score(net, mu, sd, F, I)


if __name__ == "__main__":
    main()
