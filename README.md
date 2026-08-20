# GPT from Scratch

## 02 — Bigram language model

Character-level bigram, fit two independent ways:
**counting** (closed-form MLE) and **gradient descent** (softmax regression).
Both converge to the same 3.54 bpc — verified by a test, not by eyeballing.

Beyond the lecture:
- 90/10 word-level train/val split (the lecture has none)
- Smoothing sweep across three training-set sizes, showing that the
  bias–variance U-curve is a small-data phenomenon: it disappears once
  0% of validation bigrams are unseen in training
- Unigram baseline decomposing the 1.22 bpc gain into zero-order
  (0.70, letter frequency) and first-order (0.52, adjacent correlation) parts

See [`experiments/bpc.md`](experiments/bpc.md).
