# Bits per character across models

All losses are cross-entropy converted to bits: `bpc = nats / ln 2`.
Split: 90/10 by **word** (not by bigram — same name's characters stay together):
28,829 train words / 3,204 val words → 205,579 train bigrams / 22,567 val bigrams.

| model | context | params | train bpc | val bpc | notes |
|---|---|---|---|---|---|
| uniform | 0 | 0 | 4.7549 | 4.7549 | log2(27), theoretical |
| unigram | 0 | 27 | 4.0742 | 4.0558 | letter frequencies only |
| bigram (counting, k=1) | 1 | 729 | 3.5421 | 3.5362 | |
| bigram (counting, k=0.001) | 1 | 729 | 3.5412 | 3.5354 | best on the sweep grid, see below |
| bigram (neural, λ=0.01) | 1 | 729 | 3.5536 | 3.5479 | 300 steps, lr 50 |
| bigram (neural, λ=0) | 1 | 729 | 3.5487 | 3.5433 | 0.0075 bpc above the counting MLE |

## Findings

- Bigram saves **1.219 bpc** over uniform on validation. The unigram row splits
  this in two: **0.699** from letter frequency (zero-order) and **0.520** from
  adjacent-character correlation (first-order mutual information).
- Counting and gradient descent reach the same optimum — the objective is convex
  (multinomial logistic regression), so there is only one. Unregularized GD lands
  0.0075 bpc above the closed-form MLE floor of 3.5412 after 300 steps, and the
  test suite asserts it can never land *below* it.
- **The bias–variance U-curve is a small-data phenomenon.** On the full training
  set, smoothing has no variance to correct: 0.00% of validation bigrams are
  unseen in training, so validation loss is monotonic in `k` and the best `k`
  sits at the left edge of the grid. Starve the training set and the U appears:

  | train words | val bigrams unseen in train | best k | val bpc |
  |---|---|---|---|
  | 500 | 2.33% | 0.259 | 3.6190 |
  | 2,000 | 0.49% | 0.259 | 3.5608 |
  | 28,829 | 0.00% | 0.001 (grid edge) | 3.5354 |

  The 102 zero cells in the full count matrix (e.g. `qx`) are pairs English names
  never use — they are absent from validation too, which is why k→0 is safe here.
- Failure modes of the bigram, from actual samples: single-letter "names"
  (`p`, `a`), runaway length (`ksahnaauranileviasshedainrwieta`), vowel–consonant
  mush (`faveumerifontume`, `phynslenaruani`). All symptoms of first-order Markov:
  everything before the current character is discarded.

## Reproducibility

- torch: 2.6.0+cu124
- seed: 2147483647 (init & sampling), 42 (train/val split)
- Sampled names are not guaranteed to match across PyTorch versions;
  training metrics are.

## Figures

- `figures/bigram_smoothing_sweep.png`
- `figures/bigram_convergence.png`
