# Bits per character across models

All losses are cross-entropy converted to bits: `bpc = nats / ln 2`.
Split: **80/10/10 by word, seed=42. Test set untouched until the final chapter.**
Splitting by word (not by bigram) keeps a name's adjacent characters on the same
side — a bigram-level split would leak `em` into train while `mm` sits in val.

25,626 train / 3,203 val / 3,204 test words.

| model | context | params | train bpc | val bpc | notes |
|---|---|---|---|---|---|
| uniform | 0 | 0 | 4.7549 | 4.7549 | log2(27), theoretical |
| unigram (marginal) | 0 | 27 | 4.0748 | 4.0688 | best context-free predictor |
| bigram (counting, k=1) | 1 | 729 | 3.5407 | 3.5558 | |
| bigram (counting, k=0.386) | 1 | 729 | 3.5400 | **3.5555** | best k from the sweep |
| bigram (counting, k=0) | 1 | 729 | 3.5397 | — | closed-form MLE floor |
| bigram (neural, λ=0) | 1 | 729 | 3.5472 | 3.5622 | 0.0075 bpc above the MLE floor |
| bigram (neural, λ=0.01) | 1 | 729 | 3.5522 | 3.5662 | 300 steps, lr 50 |
| MLP (block_size=1) | 1 | 7,897 | 3.5450 | 3.5609 | sanity check: rediscovers the bigram |
| MLP (n_embd=2) | 3 | 2,297 | — | 3.2150 | the 2-D model used for the embedding plot |
| MLP (block_size=3) | 3 | 11,897 | 2.9643 | 3.0648 | the lecture's configuration |
| MLP (block_size=6) | 6 | 17,897 | 2.8197 | **2.9769** | best so far |
| MLP (block_size=8) | 8 | 21,897 | 2.8132 | 2.9870 | train ↓ but val ↑ — overfitting |

MLP rows: `n_embd=10`, `n_hidden=200`, 60k steps of minibatch-64 SGD, lr 0.1 → 0.01 at 40k.

## Findings — bigram

- Bigram saves **1.199 bpc** over uniform on validation. The unigram row splits
  this in two: **0.686** from letter frequency (zero-order) and **0.513** from
  adjacent-character correlation (first-order mutual information).
- Counting and gradient descent reach the same optimum — the objective is convex
  (multinomial logistic regression), so there is only one. Unregularized GD lands
  0.0075 bpc above the closed-form MLE floor of 3.5397 after 300 steps, and the
  test suite asserts it can never land *below* it.
- **The smoothing U-curve's depth is set by the unseen-bigram rate.** Smoothing
  trades two errors: too little and unseen bigrams get near-zero probability
  (loss explodes), too much and the distribution flattens toward uniform.

  | train words | val bigrams unseen in train | best k | val bpc | U depth |
  |---|---|---|---|---|
  | 500 | 2.66% | 0.386 | 3.6490 | 0.180 bpc |
  | 2,000 | 0.58% | 0.386 | 3.5806 | 0.040 bpc |
  | 25,626 (full) | 0.03% | 0.386 | 3.5555 | 0.002 bpc |

  Depth tracks the unseen rate almost linearly (~6.8 bpc per unit rate). 109 of
  the 729 count cells are zero (`qx`, `jq`, …), but those are pairs English names
  never use, so they cost nothing at validation. Only *rare-but-real* pairs matter.

## Findings — MLP

- **The MLP at `block_size=1` reproduces the bigram** (3.5609 vs 3.5555 val, a
  0.005 bpc gap from SGD not fully converging). Same information, different
  parameterisation: 7,897 weights arriving where 729 counts already were. This is
  the cleanest available check that the MLP is wired correctly.
- **Context is worth 0.58 bpc, and it saturates.** Going 1 → 3 buys 0.496 bpc,
  3 → 6 buys another 0.088, and 6 → 8 *loses* 0.010 while train loss keeps
  falling — the model starts memorising instead of generalising. On this dataset
  the useful context is about 6 characters, which is unsurprising: the median
  name is 6 letters long, so beyond that the window mostly sees padding.
- **Learning rate must be swept on a log scale.** `10**linspace(-3, 0, 1000)`
  puts 333 candidates in each decade; the naive `linspace(0.001, 1, 1000)` puts
  9 in `1e-3..1e-2`, 90 in `1e-2..1e-1`, and 900 in `1e-1..1e0` — i.e. 90% of the
  budget in the one decade most likely to diverge. Sweep found lr ≈ 0.23.

  Caveat worth stating: this one-step-per-candidate method conflates "this lr is
  good" with "we are further into training", since loss falls with step count
  regardless. It reliably locates the *divergence ceiling*, not the optimum.
- **The 2-D embedding recovers phonetic structure with no supervision.** All five
  vowels land in one quadrant, `y` (a semivowel) sits next to them, and `.` is
  isolated far from every letter — it is the only token whose successor
  distribution is "nothing". Rare letters (`q`, `j`, `x`, `z`) stay bunched near
  the origin: few gradients, so they barely moved from initialisation.
- Sampled names at `block_size=3`: `mora`, `kayah`, `see`, `eliah`, `malaia`,
  `kylene`, `salynn` — plausible. Failures are now long-range rather than local:
  `miloparekelseananaraelyn`, `kyriquopoof`. Three characters of context cannot
  track how long the name already is.

## Reproducibility

- torch 2.6.0+cu124, Python 3.11.15
- seed 42 (train/val/test split, `nnzh.data.SPLIT_SEED` — never change it, every
  number in this table depends on it), 2147483647 (init, minibatch, sampling)
- Sampled names are not guaranteed to match across PyTorch versions;
  training metrics are.

## Figures

- `figures/bigram_smoothing_sweep.png`
- `figures/bigram_convergence.png`
- `figures/mlp_lr_sweep.png`
- `figures/mlp_block_size.png`
- `figures/mlp_embeddings.png`
