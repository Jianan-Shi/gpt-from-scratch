# GPT from Scratch

Working through Karpathy's *Neural Networks: Zero to Hero*, one directory per
lecture, with a shared dataset and a shared train/val/test split so the loss
numbers stay comparable across chapters.

```
data/names.txt        shared by chapters 02–06
nnzh/                 shared utilities (vocab, split, dataset, bpc) — the split seed lives here
01_micrograd/         autograd engine + PyTorch gradient parity tests
02_bigram/            bigram LM, fit by counting and by gradient descent
03_mlp/               Bengio-style MLP with a fixed context window
04_batchnorm/         (in progress) initialisation, activation stats, BatchNorm
experiments/bpc.md    the running scoreboard: bits per character, every model
figures/
```

Setup:

```bash
pip install -e .      # editable install of nnzh/, so `from nnzh.data import ...` works anywhere
pytest                # 23 tests across all chapters
```

## 02 — Bigram language model

Character-level bigram, fit two independent ways:
**counting** (closed-form MLE) and **gradient descent** (softmax regression).
Both reach the same optimum, 0.0075 bpc apart after 300 steps — verified by a
test asserting gradient descent can never beat the closed form, not by eyeballing.

Beyond the lecture:
- 80/10/10 word-level train/val/test split (the lecture has none). Test set is
  untouched until the final chapter.
- Smoothing sweep across three training-set sizes, showing that the bias–variance
  U-curve's *depth* is set by the unseen-bigram rate: 2.66% unseen → 0.180 bpc
  deep, 0.03% unseen → 0.002 bpc deep, i.e. effectively flat on full data.
- Unigram baseline decomposing the 1.199 bpc gain over uniform into zero-order
  (0.686, letter frequency) and first-order (0.513, adjacent correlation) parts.

See [`experiments/bpc.md`](experiments/bpc.md).

## 03 — MLP with a context window

Bengio-style MLP: `block_size` characters embedded into `n_embd` dimensions,
concatenated, one tanh hidden layer, softmax over 27 characters. 11,897 params
at the lecture's configuration, **3.0648 bpc val** — 0.49 bpc better than the
bigram.

Beyond the lecture:
- **`block_size=1` reproduces the bigram** (3.5609 vs 3.5555 val). A neural net
  with 7,897 weights lands where 729 counts already were — the cheapest available
  proof that the wiring is right.
- Context sweep from 1 to 8 characters showing where it saturates: 1→3 buys
  0.496 bpc, 3→6 buys 0.088, and 6→8 *costs* 0.010 while train loss keeps
  falling. Useful context on this dataset is ~6 characters.
- Learning-rate sweep plotted alongside the sampling grids themselves, showing
  why `10**linspace(-3, 0)` (333 candidates per decade) beats
  `linspace(0.001, 1)` (9 / 90 / 900) — and noting that the one-step-per-candidate
  method locates the divergence ceiling, not the optimum.
- 2-D embedding plot: all five vowels cluster, `y` joins them, `.` is isolated —
  phonetic structure learned with no supervision.

Tests assert the initialisation puts step-0 loss at `ln 27`, that the flattened
embedding preserves position order, and that the context window resets between
words (a leak that would otherwise be silent).

See [`experiments/bpc.md`](experiments/bpc.md).
