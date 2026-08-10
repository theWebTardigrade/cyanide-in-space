# Libraries
import numpy as np
import matplotlib.pyplot as plt
import linmix
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"

# Import Data
def read_table_as_dict(filename):
    with open(filename) as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith('#')]

    headers = [h.strip() for h in lines[0].split(',')]
    table = {h: [] for h in headers}

    for line in lines[1:]:
        values = line.strip().split(',')
        for h, v in zip(headers, values):
            v = v.strip()
            if not v or v.lower() == 'nan':
                value = np.nan
            else:
                try:
                    value = float(v)
                except ValueError:
                    value = v
            table[h].append(value)

    return table

from astropy.cosmology import Planck18