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
04_batchnorm/         initialisation, activation stats, BatchNorm, diagnostics
05_backprop/          (in progress) backprop by hand through the whole net
experiments/bpc.md    the running scoreboard: bits per character, every model
figures/
```

Setup:

```bash
pip install -e .      # editable install of nnzh/, so `from nnzh.data import ...` works anywhere
pytest                # 42 tests across all chapters
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

## 04 — Initialisation, activation statistics, BatchNorm

Six layers (five tanh hidden layers of 100, plus an output layer), 47,024 params,
**3.0521 bpc val** — indistinguishable from the 03 MLP's 3.0648 with a fifth of the
context budget spent on depth instead. That *is* the chapter's result: none of part
3's machinery moves the score on this dataset. What it moves is how much you have to
care about getting the initialisation right.

Beyond the lecture:
- **Ablation, one fix at a time** (squashed output layer → Kaiming gain 5/3 →
  BatchNorm) under an identical 100k-step budget. All five variants land within
  0.02 bpc of each other, and the ordering runs the wrong way: each added fix costs a
  hair of final loss. Three seeds per configuration confirm the BatchNorm gap
  (0.033 bpc) is real and not seed noise, while the initialisation gaps are not.
- **BatchNorm makes the forward pass exactly scale-invariant.** `gain=1` and
  `gain=5/3` give pointwise identical logits (maxdiff 3.5e-6); without BN the same
  pair differs by 0.15. This is what "BN removes the need to tune the initialisation"
  means literally, and a test asserts it.
- **What forgetting `model.eval()` actually costs**, measured: 3.0452 bpc at batch
  256, 3.1205 at batch 32, 4.6392 at batch 2 — the last is worse than the bigram.
  BN couples the examples in a batch, and at inference that becomes a dependency on
  who else happened to be in the batch. Nothing raises an error.
- **Depth × BatchNorm sweep** (2/4/6/10 layers): depth saturates after 4 layers
  (4 → 10 buys 0.010 bpc), and the BN branch is ~0.03 bpc worse at every depth. With
  Kaiming initialisation already in place, a 10-layer tanh net trains fine without BN.
- **The update:data ratio disagrees with the lecture's learning rate.** lr=0.1 sits at
  log10 ≈ −2.55 against the −3 rule of thumb, pointing at lr ≈ 0.03. The diagnostic
  costs 1000 steps instead of a full sweep.

Tests assert that a bias in front of BatchNorm has literally no effect (hence
`bias=False`), that the running buffers agree with an explicit calibration pass over
the training set, that eval mode decouples examples inside a batch while train mode
does not, and that train mode is undefined at batch 1 — which is why sampling has to
switch to eval.

See [`experiments/bpc.md`](experiments/bpc.md).

## 05 — Becoming a backprop ninja

The part-3 network again — one 200-unit hidden layer with BatchNorm, 12,297 params,
**3.0629 bpc val** against that model's 3.0648. Landing in the same place is the
result: this chapter replaces `loss.backward()`, not the network. All 200k steps run
under `torch.no_grad()`, driven by gradients derived by hand — through cross-entropy,
tanh, BatchNorm's three paths, the matmuls, and the embedding table's scatter-add.

The four exercises go from mechanical to closed-form: reproduce every intermediate
gradient one op at a time, then collapse `softmax` + NLL into `(p - onehot) / n`, then
collapse the six BatchNorm steps into one expression, then train on the result.

Beyond the lecture:
- **`exact: False` from `hpreact` down is the kernel, not the derivation.** Sixteen
  nodes fail bit-equality while passing `allclose`, which reads like a broken formula.
  It is one 1-ULP difference at `tanh` propagating: torch 2.6's vectorised
  `tanh_backward` evaluates `1 - h*h` with an FMA, rounding once where
  `(1.0 - h**2) * dh` rounds twice — 626 of 2048 elements differ in the last bit.
  Seeding `dhpreact` from `aten::tanh_backward` makes all fifteen downstream nodes
  `exact: True` again, which is the proof that the hand-written chain is bit-perfect.
  The same code is fully exact on torch 2.11's CPU build. `approximate` is the
  criterion that means anything here.
- **The guide's parameter cell was not reproducible.** `bngain` and `bnbias` were
  built without `generator=g`, so they drew from the unseeded global RNG and every
  re-execution produced different losses, different gradients, and a different
  reported bpc. Inherited from the upstream notebook, and invisible until you compare
  two runs. Fixed in both the exercise cell and the training cell.

Exercises 2 and 3 are expected to fail bit-equality on their own terms: a fused
closed-form expression cannot be expected to round like an eight-step chain
(`maxdiff` 6.1e-9 and 9.3e-10 respectively).

See [`experiments/bpc.md`](experiments/bpc.md).
