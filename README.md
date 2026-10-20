# Grouped cross-validation for well-log facies classification

How much does a random train/test split flatter a model when the rows are not
independent? This project answers that for one public dataset. It trains
classifiers to predict rock type (lithofacies) from wireline logs, then scores
them twice: once with shuffled K-fold, and once by holding out **entire wells**.
Holding out wells matches how the model would actually be used.

**Headline:** with random K-fold, LightGBM looks like a 0.71 macro-F1 model and
clearly better than logistic regression. When each well is held out, it scores
0.38 and is **no better than logistic regression**. Both still beat the
majority-class baseline by a wide margin. Leakage inflated the score, and it
would also have led us to pick the wrong model.

```
python run.py      # ~1 minute on a laptop; writes every table below to results/
pytest             # 26 tests, ~10 s
```

---

## 1. The problem

A geologist describes rock types ("facies") from core samples, but coring is
expensive and only a few wells get cored. Every well gets cheap wireline logs
(gamma ray, resistivity, porosity…). The goal is to predict the facies in
**a new, uncored well** from its logs.

So the unit we need to generalise to is the **well**, not the row. Rows are
half-foot depth samples, and neighbouring samples in the same well are nearly
identical in both their logs and their label. If a random split puts sample
*n* in test and samples *n−1* and *n+1* in training, the model can score well
by recognising the neighbours instead of learning anything about rocks.

## 2. Data

[SEG 2016 machine-learning contest](https://github.com/seg/2016-ml-contest)
training file, **Apache-2.0**, redistributed unmodified in `data/` with its
license. It holds wells from the Hugoton and Panoma fields, Kansas. Full
provenance, column dictionary and checksum: [`data/README.md`](data/README.md).

| Step | Rows | Wells | Why |
|---|---:|---:|---|
| Raw file | 4,149 | 10 | |
| Drop rows with no `PE` log | 3,232 | 8 | Two wells have no PE log at all. Imputing it would mean inventing a whole log for those wells. |
| Drop `Recruit F9` | **3,164** | **7** | Not a real well: the organisers added hand-picked examples of the rare facies 9. As a "held-out well" it is meaningless. |

9 classes, imbalanced: the most common class (non-marine coarse siltstone) is
23.3% of rows, the rarest (bafflestone) 2.9%. One facies "bed" spans a
**median of 4 consecutive samples** (mean 6.3). That is why neighbours share
labels.

Features (7): five logs (`GR`, `ILD_log10`, `DeltaPHI`, `PHIND`, `PE`) and two
geological indicators supplied with the data (`NM_M` non-marine/marine flag,
`RELPOS` relative position in the formation). `Depth` is deliberately **not**
used: absolute depth identifies a well, not a rock type.

## 3. Method

| | |
|---|---|
| **Baselines** | Majority class (`DummyClassifier`), then standardised multinomial **logistic regression**. |
| **Main model** | **LightGBM** (300 trees, lr 0.05, 15 leaves, row/column subsampling 0.8). Hyper-parameters were fixed before the comparison and not tuned (see Limitations). |
| **Leaky CV** | `KFold(n_splits=7, shuffle=True)`: rows are shuffled, so every test fold contains rows from every well. |
| **Honest CV** | `GroupKFold(n_splits=7)` grouped by well, i.e. **leave-one-well-out**. A test check (`assert_no_group_overlap`) fails if any well appears in both train and test. |
| **Same budget** | Both schemes use 7 folds, so each model trains on ~6/7 of the rows either way. The only difference is *which* rows are held out. |
| **Metrics** | Pooled out-of-fold **accuracy** (= micro-F1, the contest metric), **macro-F1** (treats rare facies equally), **adjacent accuracy** (a prediction counts if it is the true facies or a geologically adjacent one, using the contest's published adjacency table). |
| **Extra feature set** | `base+window`: per-well first differences and centred rolling means (5 and 15 samples) of each log. These are computed within a well only, never across wells. |

## 4. Results

All numbers come from `python run.py` (pinned versions in `requirements.txt`).
The full tables are in [`results/results.md`](results/results.md).

### Baseline vs model, under each CV scheme

| Features | Model | K-fold acc | K-fold macro-F1 | **Well-out acc** | **Well-out macro-F1** | Well-out adj. acc | Well-out acc, worst–best well |
|---|---|---:|---:|---:|---:|---:|---:|
| base | majority | 0.233 | 0.042 | **0.233** | **0.042** | 0.509 | 0.139–0.284 |
| base | logistic | 0.568 | 0.497 | **0.494** | **0.379** | 0.895 | 0.443–0.555 |
| base | LightGBM | 0.716 | 0.711 | **0.478** | **0.380** | 0.897 | 0.349–0.590 |
| base+window | logistic | 0.599 | 0.565 | **0.491** | **0.391** | 0.897 | 0.415–0.597 |
| base+window | LightGBM | 0.800 | 0.795 | **0.485** | **0.382** | 0.906 | 0.383–0.622 |

### How much random K-fold inflates the score

| Features | Model | Macro-F1 inflation (K-fold − well-out) | Accuracy inflation |
|---|---|---:|---:|
| base | majority | 0.000 | 0.000 |
| base | logistic | +0.118 | +0.074 |
| base | LightGBM | **+0.332** | **+0.238** |
| base+window | logistic | +0.175 | +0.108 |
| base+window | LightGBM | **+0.413** | **+0.315** |

### What this shows

1. **The models do learn something real.** On wells they have never seen, both
   models reach ~0.49 accuracy and ~0.38 macro-F1, against 0.23 / 0.04 for
   the majority class. ~80% of LightGBM's errors are a geologically adjacent facies.
2. **Random K-fold overstates LightGBM's macro-F1 by 0.33 (0.71 vs 0.38).**
   The more flexible the model, the more it gains from the leak. The majority
   baseline cannot exploit neighbours at all, so its inflation is exactly 0.
3. **Leakage would have changed the model choice.** Under K-fold, LightGBM
   beats logistic regression by +0.21 macro-F1. Held out by well, the two tie
   (0.380 vs 0.379). Per well, LightGBM wins on 3, loses on 3 and ties on 1.
4. **For LightGBM, window features only help on the leaky split.** They add +0.084
   macro-F1 to LightGBM under K-fold and +0.002 under well-out CV. Smoothing
   makes neighbouring rows even more alike, so it mostly amplifies the leak.
   For logistic regression they lift well-out macro-F1 from 0.379 to 0.391.
   That is the best well-out macro-F1 in the table, but with 7 wells it is
   not distinguishable from noise.

### The leak, made explicit

`neighbour_copy_predictions` is a "model" that uses **no logs at all**. For
each test row it copies the label of the nearest-in-depth training row from
the same well.

| CV | Test rows that have a same-well neighbour in training | Accuracy of copying its label |
|---|---:|---:|
| random K-fold | 100% | **0.842** |
| leave-one-well-out | 0% | not possible |

Under random K-fold this label-copier beats every real model in the table.
That is the leak: the split is testing interpolation within a well. It is not
testing prediction in a new one.

## 5. Feature importance (held-out wells)

There are two views for the well-out LightGBM models. Gain share is how much
each feature was used for splits (measured on training data). Permutation
drop is how much accuracy falls on the **held-out well** when that feature is
shuffled (mean over the 7 wells).

| Feature | Gain share | Permutation accuracy drop | ± across wells |
|---|---:|---:|---:|
| `NM_M` | 0.218 | **0.178** | 0.034 |
| `PHIND` | 0.134 | 0.076 | 0.036 |
| `GR` | 0.152 | 0.048 | 0.010 |
| `ILD_log10` | 0.138 | 0.031 | 0.021 |
| `RELPOS` | 0.127 | 0.030 | 0.021 |
| `PE` | 0.127 | 0.021 | 0.025 |
| `DeltaPHI` | 0.103 | 0.018 | 0.015 |

`NM_M` dominates, and that needs a caveat. It agrees with the facies' own
marine/non-marine split (facies 1–3 vs 4–9) on 98% of rows. So it carries
half the answer, and it is itself an interpretation rather than a raw
measurement (see Limitations). The two importance views also disagree in
places: `GR` ranks second by gain but third by permutation. Gain importance
alone would have overstated it.

## 6. Error analysis (well-out LightGBM)

* **Errors stay inside the depositional setting.** Only 3.9% of errors cross
  the marine/non-marine boundary. The model rarely calls a river sandstone a
  marine limestone. It struggles to tell apart neighbours within a group.
* **Top confusions:** fine siltstone→coarse siltstone (234 rows), sandstone→coarse
  siltstone (201), coarse→fine siltstone (198), wackestone→packstone (128),
  mudstone→wackestone (122).
* **Weakest classes (recall):** bafflestone 0.08, mudstone 0.10, sandstone 0.12,
  dolomite 0.26. Three are small classes. Sandstone is mostly absorbed into
  the much larger coarse-siltstone class.
* **Wells differ a lot:** accuracy ranges from 0.35 (CROSS H CATTLE) to 0.59
  (LUKE G U). With only 7 wells, a single "well-out score" hides this spread,
  so the per-well table is in `results/per_well.csv`.

## 7. Limitations and things not done

Stated plainly, because this README should not claim more than the data supports:

* **Only 7 wells.** Each fold score comes from a single well, and the spread is
  wide (0.35–0.59). The pooled well-out score is an estimate with real
  uncertainty. I report the fold range, not a confidence interval.
* **One field, one basin.** Nothing here says how the model would do in
  another basin, or with other logging tools or vintages.
* **`NM_M` and `RELPOS` are interpretations.** A geologist derives them from
  formation tops and depositional setting. In a truly new well someone would
  have to supply them first. `NM_M` alone accounts for most of the
  permutation importance. A logs-only model would score lower, and I did not
  run that ablation.
* **No hyper-parameter tuning.** Honest tuning needs *nested* grouped CV: an
  inner well-out loop for tuning inside an outer one for scoring. With 7 wells
  that leaves 6 for the inner loop. I fixed sensible defaults instead. A tuned
  LightGBM might pull ahead of logistic regression, but this project does not
  show that.
* **Two wells dropped** for missing `PE` (905 rows). A model without `PE`, or
  with `PE` imputed from other logs, could use all 9 real wells. Not done.
* **Class imbalance not addressed.** There are no class weights or resampling,
  which is part of why the rare facies have such low recall.
* **No depth-sequence modelling.** Each sample is classified on its own.
  Smoothing predictions down the well (median filter, HMM/CRF) is a known
  improvement for this kind of data, and was not tried.
* **Not comparable to the contest leaderboard.** The winners scored ~0.64
  micro-F1 on two blind wells. They trained on more wells, used heavy feature
  engineering and ensembles, and were scored on different test wells. My 0.48
  is a leave-one-well-out score on 7 training wells with an untuned model.
* **Labels are not ground truth.** Facies come from core descriptions, and
  adjacent facies grade into each other. Some "errors" are boundary judgement
  calls. Adjacent accuracy (~0.90) is reported for that reason.
* **Sampling is not perfectly regular.** After cleaning there are 20 depth gaps
  of 1–5.5 ft within wells and 3 duplicated (well, depth) rows. Window features
  count samples, not feet, so a few windows span a gap. This does not affect
  the leakage comparison.
* **Only one leaky scheme compared.** Blocked K-fold within wells, which
  holds out contiguous depth intervals, would sit between the two. CatBoost
  was not tried either.

## 8. Why this generalises beyond geology

Any dataset with repeated rows per entity has the same trap: several visits
per patient, days per store, sessions per user, readings per sensor. If the
deployment question is "how will this do on a *new* patient/store/well?", the
split has to hold out whole entities. Otherwise the CV score measures memory
of entities you already have.

## Project layout

```
run.py                  one command: data → CV grid → importance → error analysis → results/
facies_cv/
  config.py             paths, columns, facies names, adjacency table
  data.py               checksum-verified loading, cleaning, bed-run statistics
  features.py           per-well depth-window features
  cv.py                 K-fold / grouped splits, group-overlap check, neighbour-copy probe
  metrics.py            accuracy, macro-F1, adjacent accuracy
  models.py             majority, logistic regression, LightGBM factories
  evaluate.py           out-of-fold predictions for every (features, model, CV) cell
  analysis.py           gain + permutation importance, per-well, confusion, error summary
tests/                  26 pytest tests (data, CV splits, metrics, features, end-to-end smoke run)
results/                generated tables (committed so the README numbers are traceable)
data/                   facies_vectors.csv + its Apache-2.0 license
.github/workflows/      CI: pip install + pytest on ubuntu, Python 3.10
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py           # full run, writes results/
python run.py --quick   # 30 trees instead of 300, for a fast smoke check
pytest -v
```

Results were produced with Python 3.10 on Windows using the pinned versions.
LightGBM runs single-threaded with `deterministic=True`, but the last decimal
place may still differ on other platforms.

## References

* Hall, B. (2016). Facies classification using machine learning. *The Leading Edge*, 35(10), 906–909.
* Dubois, M. K., Bohling, G. C., & Chakrabarti, S. (2007). Comparison of four approaches to a rock facies classification problem. *Computers & Geosciences*, 33(5), 599–617.
* Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.

---

Code: MIT (see `LICENSE`). Data: Apache-2.0, © the SEG 2016 contest
contributors (see `data/`). Project by Ziwei Su.
AI-assisted build (with Claude).
