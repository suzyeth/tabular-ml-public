"""Constants shared across the pipeline: paths, column names, facies metadata."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "facies_vectors.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/seg/2016-ml-contest/master/facies_vectors.csv"
)
DATA_SHA256 = "83fd9d15bbab3aed0c1dca20e7c7d2befedd110b04ee21f8c790ac03d70aa70b"

TARGET = "Facies"
GROUP = "Well Name"
DEPTH = "Depth"

# Wireline logs (continuous measurements down the borehole).
LOG_COLUMNS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
# Geological indicators supplied with the data: non-marine/marine flag and
# relative position within the formation.
INDICATOR_COLUMNS = ["NM_M", "RELPOS"]
BASE_FEATURES = LOG_COLUMNS + INDICATOR_COLUMNS

EXPECTED_COLUMNS = [TARGET, "Formation", GROUP, DEPTH] + BASE_FEATURES

# "Recruit F9" is not a real well: the contest organisers added it as hand-picked
# examples of the rare facies 9. Treating it as a held-out "well" is meaningless.
PSEUDO_WELLS = ["Recruit F9"]

FACIES_NAMES = {
    1: "SS",    # non-marine sandstone
    2: "CSiS",  # non-marine coarse siltstone
    3: "FSiS",  # non-marine fine siltstone
    4: "SiSh",  # marine siltstone and shale
    5: "MS",    # mudstone (limestone)
    6: "WS",    # wackestone (limestone)
    7: "D",     # dolomite
    8: "PS",    # packstone-grainstone (limestone)
    9: "BS",    # phylloid-algal bafflestone (limestone)
}

# Facies 1-3 were deposited on land / in rivers, 4-9 in the sea (the NM_M flag).
NON_MARINE_FACIES = (1, 2, 3)

# Facies that grade into each other in the rock record. Taken from the contest's
# tutorial notebook (Hall, 2016), converted from 0-based to 1-based labels.
# Kept exactly as published, including its one asymmetry: 9 lists 7, 7 does not list 9.
ADJACENT_FACIES = {
    1: (2,),
    2: (1, 3),
    3: (2,),
    4: (5,),
    5: (4, 6),
    6: (5, 7, 8),
    7: (6, 8),
    8: (6, 7, 9),
    9: (7, 8),
}

RANDOM_STATE = 42
WINDOW_SIZES = (5, 15)  # in samples, not feet: nominal spacing is 0.5 ft, with a few gaps
