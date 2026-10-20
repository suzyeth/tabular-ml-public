# Data

`facies_vectors.csv` — well-log training data from the SEG 2016 *Geophysical
Tutorial* machine-learning contest.

| | |
|---|---|
| Source | <https://github.com/seg/2016-ml-contest> (file `facies_vectors.csv`, repository last commit `160430c`, Feb 2017) |
| License | Apache License 2.0 — full text in [`LICENSE-Apache-2.0.txt`](LICENSE-Apache-2.0.txt). Redistributed here unmodified. |
| Original data | Wireline logs and core-described facies from wells in the Hugoton and Panoma gas fields, Kansas (Kansas Geological Survey; Dubois, Bohling & Chakrabarti, 2007). |
| Tutorial | Hall, B. (2016). *Facies classification using machine learning.* The Leading Edge 35(10), 906–909. doi:10.1190/tle35100906.1 |
| SHA-256 | `83fd9d15bbab3aed0c1dca20e7c7d2befedd110b04ee21f8c790ac03d70aa70b` (checked by `facies_cv.data.load_raw`) |

4,149 rows × 11 columns, one row per half-foot depth sample:

| Column | Meaning |
|---|---|
| `Facies` | Target: lithofacies class 1–9, described from core by geologists |
| `Formation`, `Well Name`, `Depth` | Identifiers; `Well Name` is the CV group |
| `GR` | Gamma ray |
| `ILD_log10` | Resistivity (log10) |
| `DeltaPHI` | Neutron–density porosity difference |
| `PHIND` | Average neutron–density porosity |
| `PE` | Photoelectric effect (missing for two wells) |
| `NM_M` | Non-marine (1) / marine (2) indicator |
| `RELPOS` | Relative position within the formation |

If the file is missing, `python -m facies_cv.data` downloads it again.
