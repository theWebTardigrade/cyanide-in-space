# Libraries
import numpy as np
import math
import matplotlib as ml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import csv

from scipy.stats import linregress
import pandas as pd

import linmix
from scipy.stats import chi2

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

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


#### GAO & SOLOMON 2004
Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/table_Gao2004_corr.data')
GS2004_LFIR, GS2004_LHCN = np.genfromtxt(Table, unpack=True, usecols=(2,4))
Table.close()

GS2004_LFIR = GS2004_LFIR*1e10
GS2004_LHCN = GS2004_LHCN*1e8

#### GAO 2007
# Detections
G2007_FIR = np.array((17, 0.25, 5.0, 3.4, 0.93))
G2007_HCN = np.array((6.5, 0.25, 3.0, 1.2, 0.6))
G2007_CO = np.array((74, 0.92, 37, 6.5, 3.7))

G2007_FIR = G2007_FIR*1e12
G2007_HCN = G2007_HCN*1e9
G2007_CO = G2007_CO*1e9


#### KRIPS 2008
# Krips+2008
Krips_LFIR = np.array((10.8136,9.49601,9.14275,9.22923,9.58508,9.45516,10.6883,10.1939,11.9754))
Krips_LHCN = np.array((8.15425,6.95629,6.442,6.51264,7.21742,6.79601,7.38231,6.845,9.05306))


#### Garcia-Burillo+2012
Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/table_ULIRGS_GB2012b.data')
GB2012_LFIR, GB2012_LCO, GB2012_LHCN = np.genfromtxt(Table, unpack=True, usecols=(2, 3, 4))
Table.close()
GB2012_LFIR = GB2012_LFIR*1e11

#### Privon 2015
Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/table_Privon2015.data')
P2015_LFIR, P2015_LHCN = np.genfromtxt(Table, unpack=True, usecols=(1, 2))
Table.close()
P2015_LFIR = 10**P2015_LFIR
P2015_LHCN = 10**P2015_LHCN


#############################################################################
#############################################################################

# Create the figure

fig = plt.figure()
ax=fig.add_subplot(111)
plt.xlabel(r"$L'_{\mathrm{HCN(1-0)}}$ [K km s$^{-1}$ pc$^2$]", fontsize=16)
plt.ylabel(r'$L_\text{FIR}$ [L$_\odot$]', fontsize=16)


plt.tick_params(axis='both', which = 'major', length=10, direction = 'in', width=0.5, color = 'black', labelsize = 13, top=True, right=True)
plt.tick_params(axis='both', which = 'minor',length=5, direction = 'in', width=0.5, color = 'black', labelsize = 13, top=True, right=True)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e6, 1e12)
plt.ylim(1e9, 1e15)

def UL(x, y, color, label, edgecolor=None, size=None, zorder=5, marker_new=None):
    plt.plot([0.70*x, x], [y, y], c = edgecolor, lw=2 , zorder = zorder)
    plt.scatter([x], [y], facecolor=color, edgecolor = 'black',s=size, zorder=zorder, label= label, marker=marker_new)
    plt.scatter([0.70*x], [y], c = color, edgecolor = 'black', s=30, marker = '<', zorder = zorder)


# Color Pallete
color_palette=['#4285f4', '#D72000FF','#2CA030FF', '#EE6100FF', '#FFAD0AFF', '#1BB6AFFF', '#9093A2FF', '#132157FF', '#9900ff']


###########################################################################################################################
#### GAO & SOLOMON 2004 #### z=0
plt.scatter(GS2004_LHCN, GS2004_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Krips 2008 #### z=0
plt.scatter(10.0**Krips_LHCN, 10.0**Krips_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Garcia-Burillo 2012 #### z=0
plt.scatter(GB2012_LHCN, GB2012_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Privon 2015 #### z=0
plt.scatter(P2015_LHCN, P2015_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)
###########################################################################################################################



#### GAO 2007 #### DSFGs
UL(0.6e9, 0.93e12, label = 'DSFGs', color='black', edgecolor='black', size=40) # J16359 6612(B)
UL(28e9, 22e12, label = 'DSFGs', color='black', edgecolor='black', size=40) #J04135+10277
UL(46e9, 28e12, label = 'DSFGs', color='black', edgecolor='black', size=40) # SMM J02399 0136
UL(3.7e9, 6.1e12, label = 'DSFGs', color='black', edgecolor='black', size=40) #SMM J02396+0134
UL(1.6e9, 1.5e12, label = 'DSFGs', color='black', edgecolor='black', size=40) # J14011+0252; Carilli+2005


# QUASARS ####
# Upper limits
# UL(39e9, 55e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #BR 1202 0725
# UL(9.3e9, 20e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #SDSS J1148+5251
# UL(0.6e9, 2.1e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #J0911+0551
# UL(0.9e9, 2.7e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #MG0751 2716

# Detections
plt.scatter(3.2e9, 6.2e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # Cloverleaf
plt.scatter(1.2e9, 3.4e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # IRAS F10214+4724
plt.scatter(6.7e9, 17e12, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # VCV J1409+5628


### RYBAK 2026
rybak_color = '#4285f4'
rybak_size = 7
plt.errorbar(380e8/8.8, 66e12/8.8, xerr=130e8/8.8, color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.9
plt.errorbar((170e8/(0.59*10.9)), 64e12/10.9, xerr=25e8/(0.59*10.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.11
plt.errorbar((94e8/(0.59*15.3)), 66e12/15.3, xerr=19e8/(0.59*15.3), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')#G09v1.40
plt.errorbar((126e8/(0.41*4.9)), 71e12/4.9, xerr=16e8/(0.41*4.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.17
UL((74e8/(0.41*1.8)), 28e12/1.8, color = rybak_color, label = 'Rybak 2026', size=40, marker_new='s') #G15v2.235
plt.errorbar((560e8/(0.59*14.7)), 133e12/14.7, xerr=19e8/(0.59*14.7), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #J0209
plt.errorbar((52e8/0.41), 40e12, xerr=13e8/(0.41),color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #G09v1.326
plt.errorbar((52e8/(0.41*8.6)), 32e12/8.6, xerr=23e8/(0.41*8.6), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.130
plt.errorbar((180e8/(0.41*4.1)), 75e12/4.1, xerr=30e8/(0.41*4.1), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #NAv1.195
UL(3.7e10/18.2, 43e12/18.2, label = 'Rybak 2026', color=rybak_color, edgecolor=rybak_color, size=40, marker_new='s') #SDP.81
plt.errorbar((210e8/(0.41*9.2)), 90e12/9.2, xerr=15e8/(0.41*9.2), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #G12v2.43

### 2022
UL(5.1e10/5.3, 36e12/5.3, color = rybak_color, label = 'Rybak 2026', size=35, marker_new='s') #HXMM.02
UL(11.4e10/15.4, 156e12/15.4, color = rybak_color, label = 'Rybak 2026', size=35, marker_new='s') #J1609

plt.errorbar((7.60e10/(0.41*22)), 2.65e14/22, xerr=2.10e10/(0.41*22), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
# NCv1.143
plt.errorbar((3.50e10/(0.41*9.2)), 1.40e14/12.2, xerr=2.30e09/(0.41*9.2), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
# APM
plt.errorbar((5.40e10/(0.41*3)), 1.8e14/3, xerr=15e8/(0.41*3), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')



# # ### BETH WESTOBY 2026
# beth_color = '#EE6100FF'
# beth_size = 15
# r21 = 0.7
# plt.errorbar(5.90e10/(r21*5.5), 1.87e13, xerr=6.75e9/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0314-44
# plt.errorbar(5.09e10/(r21*21), 4.68e12, xerr=7.71e9/(r21*21), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2134-50
# plt.errorbar(5.58e10/(r21*10.04), 9.94e12, xerr=8.58e9/(r21*10.04), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0532-50
# plt.errorbar(6.42e10/(r21*5.5), 1.4e13, xerr=1.19e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-58
# plt.errorbar(7.85e10/(r21*5.5), 4.62e13, xerr=7.26e9/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0402-45
# plt.errorbar(2.94e10/(r21*20.1), 4.09e12, xerr=6.19e9/(r21*20.1), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0538-50
# plt.errorbar(3.04e10/(r21*5.5), 1.58e13, xerr=1.14e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2101-60
# plt.errorbar(2.09e10/(r21*5.5), 2.34e13, xerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-55
# plt.errorbar(1.04e10/(r21*5.5), 2.22e13, xerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2332-53
# plt.errorbar(1.35e10/(r21*5.49), 1.87e13, xerr=6.58e9/(r21*5.49), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0027-50 


# MPF 2026
my_color = "#9151B8"
my_edge_color = '#000000'
my_size= 15
###########
UL(5.5e10, 10**(13.73), color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # W2246

ax.text(
    5.5e10 * 0.8,      # 20% left
    2.2e14 * 1.15,     # 15% up
    "W2246",
    color=my_color,
    fontsize=10,
    ha='right',
    va='bottom'
)

###########

plt.errorbar(11.7e10/8.3, 1.77e14/8.3, xerr=3.2e10/8.3, color = my_color, mec=my_edge_color ,label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1336
plt.errorbar(29.1e10/28.2, 4.60e14/28.2, xerr=4.2e10/28.2, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J0226
plt.errorbar(9.5e10/7.6, 2.3e14/7.6, xerr=1.8e10/7.6, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1053
plt.errorbar(14.1e10/25, 8.110e13/25, xerr=2.3e10/25, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1202
plt.errorbar(5.5e10/8, 12e13/8, xerr=(1.7e10/8), color = my_color, mec=my_edge_color, label = 'This work', zorder=6, markersize=my_size, fmt='.') # J1323
UL(4.4e10/32.8, 7.4e13/32.5, color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # Eyelash
plt.errorbar(5.3e10/11.7, 10.06e13/11.7, xerr=3.3e10/11.7,  color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # Nav1.56
UL(4.38e10/6.1, 4.11e13/6.1, color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # J213+0109
plt.errorbar(3.8e10/57, 9.23e13/57, xerr=1.0e10/57, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J0116
plt.errorbar(4.2e10/11, 3.90e14/11, xerr=2.2e10/11, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # Eyebrow


########################################################################################################################
# Jimenez-Donaire 2019 trend
trend_color = '#F24F4F'

plt.plot([(10**-2.99)*1e8, (10**-2.99)*1e15], [1.e8, 1.e15], color=trend_color, lw=1, linestyle = '-')


plt.fill_betweenx(
    [1.e8, 1.e15],
    [(10**-2.99)/10**0.30*1e8, (10**-2.99)/10**0.30*1e15],
    [(10**-2.99)*10**0.30*1e8, (10**-2.99)*10**0.30*1e15],
    alpha=0.2,
    color=trend_color,
    edgecolor='none'
)





###########################################################################################################################################
### Jiménez-Donaire
literature_data = pd.read_fwf(
    "/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/apjab2b95t11_mrt.txt",
    colspecs=[
        (0, 16),   # Ref
        (17, 25),  # logLIR
        (26, 35),  # logLHCN
        (36, 37),  # Type
    ],
    skiprows=41,
    names=["Ref", "logLIR", "logLHCN", "Type"]
)

# clean whitespace
literature_data["Ref"] = literature_data["Ref"].str.strip()
literature_data["Type"] = literature_data["Type"].str.strip()

# individual galaxies
unresolved_galaxies_lit = literature_data.loc[
    ~literature_data["Type"].isin(["R"])
]
resolved_lit = literature_data.loc[
    literature_data["Type"].isin(["I"])
]

usero2015 = literature_data[literature_data["Ref"] == "Usero15"]


plt.scatter(10**usero2015['logLHCN'], 10**usero2015['logLIR'], color = '#9093A2FF', marker='.', label = 'Parts of Galaxies', zorder=5, s=20)

###########################################################################################################################################


#####
# FOR FITTING
# Coment what you don't want to include
#####

FIR_values = [
    66e12/8.8,
    64e12/10.9,
    66e12/15.3,
    71e12/4.9,
    28e12/1.8,
    133e12/14.7,
    40e12,
    32e12/8.6,
    75e12/4.1,
    43e12/18.2,
    90e12/9.2,

    36e12/5.3,
    156e12/15.4,

    # 1.87e13,
    # 4.68e12,
    # 9.94e12,
    # 1.4e13,
    # 4.62e13,
    # 4.09e12,
    # 1.58e13,
    # 2.34e13,
    # 2.22e13,
    # 1.87e13,

    #10**(13.73),
    2.2e14,
    1.77e14/8.3,
    4.60e14/28.2,
    2.3e14/7.6,
    8.110e13/25,
    12e13/8,
    7.4e13/32.5,
    10.06e13/11.7,
    4.11e13/6.1,
    9.23e13/57,
    1.1e13,

    # # Other points
    # 0.93e12,
    # 22e12,
    # 28e12,
    # 6.1e12,
    # 1.5e12,
    # 55e12,
    # 20e12,
    # 2.1e12,
    # 2.7e12,
    # 6.2e12,
    # 3.4e12,
    # 17e12,
]

HCN_values = [
    380e8/8.8,
    170e8/(0.59*10.9),
    94e8/(15.3*0.59),
    126e8/(0.41*4.9),
    74e8/(0.41*1.8),
    560e8/(0.59*14.7),
    52e8/0.41,
    52e8/(0.41*8.6),
    180e8/(0.41*4.1),
    3.7e10/18.2,
    210e8/(0.41*9.2),

    5.1e10/5.3,
    11.4e10/15.4,

    # 5.90e10/(0.7*5.5),
    # 5.09e10/(0.7*21),
    # 5.58e10/(0.7*10.04),
    # 6.42e10/(0.7*5.5),
    # 7.85e10/(0.7*5.5),
    # 2.94e10/(0.7*20.1),
    # 3.04e10/(0.7*5.5),
    # 2.09e10/(0.7*5.5),
    # 1.04e10/(0.7*5.5),
    # 1.35e10/(0.7*5.49),

    5.5e10,
    11.7e10/8.3,
    29.1e10/28.2,
    9.5e10/7.6,
    14.1e10/25,
    5.5e10/8,
    4.4e10/32.5,
    5.3e10/11.7,
    4.38e10/6.1,
    3.8e10/57,
    4.2e10/11,

    # # Other points
    # 0.6e9,
    # 28e9,
    # 46e9,
    # 3.7e9,
    # 1.6e9,
    # 39e9,
    # 9.3e9,
    # 0.6e9,
    # 0.9e9,
    # 3.2e9,
    # 1.2e9,
    # 6.7e9,
]


HCN_errors = [
    130e8/8.8,
    25e8/(0.59*10.9),
    19e8/(15.3*0.59),
    16e8/(0.41*4.9),
    0,
    19e8/(0.59*14.7),
    13e8/(0.41),
    23e8/(0.41*8.6),
    30e8/(0.41*4.1),
    0,
    15e8/(0.41*9.2),

    0,
    0,

    # 6.75e9/(r21*5.5),
    # 7.71e9/(r21*21),
    # 8.58e9/(r21*10.04),
    # 1.19e10/(r21*5.5),
    # 7.26e9/(r21*5.5),
    # 6.19e9/(r21*20.1),
    # 1.14e10/(r21*5.5),
    # 1,
    # 1,
    # 6.58e9/(r21*5.49),

    0,
    3.2e10/8.3,
    4.2e10/28.2,
    1.8e10/7.6,
    2.3e10/25,
    1.7e10/8,
    0,
    3.3e10/11.7,
    0,
    1.0e10/57,
    2.2e10/11,

    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,


]


detections = [
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,

    0,
    0,

    # 1,
    # 1,
    # 1,
    # 1,
    # 1,
    # 1,
    # 1,
    # 1,
    # 1,
    # 1,

    0,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,

    # #Other points
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 1,
    # 1,
    # 1,
    # 1,
]


################################################################################
# Define what log values to use for the fits


# Rybak2026, Westoby2026 and this work
# Already defined above

# Rybak2026, Westoby2026 and this work + z~0
all_FIR = np.concatenate([
    GS2004_LFIR,
    10.0**Krips_LFIR,
    GB2012_LFIR,
    P2015_LFIR,

    FIR_values
])

all_HCN = np.concatenate([
    np.abs(GS2004_LHCN),
    np.abs(10.0**Krips_LHCN),
    np.abs(GB2012_LHCN),
    np.abs(P2015_LHCN),

    HCN_values
])

all_errs_HCN = np.concatenate([
    np.zeros_like(GS2004_LHCN),
    np.zeros_like(10.0**Krips_LHCN),
    np.zeros_like(GB2012_LHCN),
    np.zeros_like(P2015_LHCN),

    HCN_errors
])


# 1 detection, 0 non-detections
all_detections = np.concatenate([
    np.ones_like(GS2004_LHCN),
    np.ones_like(10.0**Krips_LHCN),
    np.ones_like(GB2012_LHCN),
    np.ones_like(P2015_LHCN),

    detections

])


#### Filter for detections
FIR_values = np.asarray(FIR_values)
HCN_values = np.asarray(HCN_values)
HCN_errs = np.asarray(HCN_errors)
detections = np.asarray(detections)

mask = detections == 1

FIR_detections = FIR_values
HCN_detections = HCN_values
HCN_errs_detections = HCN_errs
detections_detections = detections


#############################################################
# What is the scatter of Rybak and Pólvora Fonseca points in comparison with JD?

print('###################################################################')

# Calculate the scatter of the points in comparison with JD/GS trend
FIR_values = np.asarray(FIR_values)
HCN_values = np.asarray(HCN_values)

log_residuals = np.log10(HCN_values) - (-2.99 + np.log10(FIR_values))
scatter_fromJD = np.std(log_residuals, ddof=1)


print('Comparing the scatter of all MR2026 and MPF2026 points to JD/GS')
print('The scatter is ', scatter_fromJD)


print('###################################################################')
# Calculate the scatter of ONLY DETECTIONS in comparison with JD/GS trend
log_residuals = np.log10(HCN_detections) - (-2.99 + np.log10(FIR_detections))
scatter_fromJD_detections = np.std(log_residuals, ddof=1)


print('Comparing the scatter of DETECTIONS of MR2026 and MPF2026 points to JD/GS')
print('The scatter is ', scatter_fromJD_detections)


print('###################################################################')
# JD intrinsic scatter
sigma_JD = 0.30 # dex


# Total uncertainty
sigma_HCN = HCN_errs_detections / (HCN_detections * np.log(10))
sigma_total = np.sqrt(sigma_HCN**2 + sigma_JD**2)

# Residuals
log_residuals = np.log10(HCN_detections) - (-2.99 + np.log10(FIR_detections))



# Chi-square
chi2_value = np.sum((log_residuals / sigma_total)**2)

dof = len(log_residuals)
reduced_chi2 = chi2_value / dof
p_value = chi2.sf(chi2_value, dof)

print(f"Chi2 = {chi2_value:.5f}")
print(f"Reduced Chi2 = {reduced_chi2:.5f}")
print(f"p-value = {p_value:.3f}")


print('###################################################################')

# ######################################################################
# Build the literature (z~0) sample restricted to LIR > 1e12 Lsun so it
# can be included in the LINMIX fit alongside Rybak2026 + this work.

lit_FIR_all = np.concatenate([
    GS2004_LFIR,
    10.0**Krips_LFIR,
    GB2012_LFIR,
    P2015_LFIR,
])

lit_HCN_all = np.concatenate([
    np.abs(GS2004_LHCN),
    np.abs(10.0**Krips_LHCN),
    np.abs(GB2012_LHCN),
    np.abs(P2015_LHCN),
])

lit_mask_highLIR = lit_FIR_all > 10**(13)

lit_FIR_highLIR = lit_FIR_all[lit_mask_highLIR]
lit_HCN_highLIR = lit_HCN_all[lit_mask_highLIR]
lit_errs_highLIR = np.zeros_like(lit_HCN_highLIR)
lit_detections_highLIR = np.ones_like(lit_HCN_highLIR)

print(f'Including {lit_mask_highLIR.sum()} literature (z~0) points with LIR>1e12 in the LINMIX fit')

# ######################################################################
#  LINMIX
# Now fitting mine + Rybak2026 points, plus the literature points
# (Gao2004, Krips2008, Garcia-Burillo2012, Privon2015) with LIR > 1e12
# To fit everything with a single line, we need to substitute the values to all

FIR_for_linmix = FIR_detections
HCN_for_linmix = HCN_detections
HCN_errs_for_linmix = HCN_errs_detections
detections_for_linmix = detections_detections


log_xvalues = np.log10(FIR_for_linmix)
log_yvalues = np.log10(HCN_for_linmix)

log_yerrs = np.array(HCN_errs_for_linmix) / (np.array(HCN_for_linmix) * np.log(10))

print('Fitting LINMIX')
linmix_model = linmix.LinMix(log_xvalues, log_yvalues, ysig=log_yerrs, delta=detections_for_linmix, seed=42)
linmix_model.run_mcmc(silent=True)

# Extract posterior medians 
alpha_chain1 = linmix_model.chain['alpha']
beta_chain1 = linmix_model.chain['beta']

alpha_med1 = np.median(alpha_chain1)
alpha_lo1, alpha_hi1 = np.percentile(alpha_chain1, [16, 84])

beta_med1 = np.median(beta_chain1)
beta_lo1, beta_hi1 = np.percentile(beta_chain1, [16, 84])

scatter1 = np.sqrt(np.median(linmix_model.chain['sigsqr']))


print(f"Best-fit intercept: {alpha_med1:.2f} (+{alpha_hi1 - alpha_med1:.2f}, -{alpha_med1 - alpha_lo1:.2f})")
print(f"Best-fit slope: {beta_med1:.2f} (+{beta_hi1 - beta_med1:.2f}, -{beta_med1 - beta_lo1:.2f})")

# Best-fit trend line
x_line = np.logspace(np.min(log_xvalues), 15, 200)
log_y_line = alpha_med1 + beta_med1 * np.log10(x_line)
y_line = 10**log_y_line

new_trend_color = '#1DBD8E'

plt.plot(
    y_line,
    x_line,
    color=new_trend_color,
    lw=1,
    linestyle='-'
)

# Intrinsic scatter region
plt.fill_betweenx(
    x_line,
    10**(log_y_line - scatter1),
    10**(log_y_line + scatter1),
    alpha=0.2,
    color=new_trend_color,
    edgecolor='none'
)


log_intercept_chain = -alpha_chain1 / beta_chain1
slope_chain = 1.0 / beta_chain1

log_intercept_med = np.median(log_intercept_chain)
log_intercept_lo, log_intercept_hi = np.percentile(log_intercept_chain, [16, 84])

slope_med = np.median(slope_chain)
slope_lo, slope_hi = np.percentile(slope_chain, [16, 84])

print(f"Best-fit slope: {slope_med:.2f} (+{slope_hi - slope_med:.2f}, -{slope_med - slope_lo:.2f})")


#############################################################################################################################################
handles, labels = ax.get_legend_handles_labels()

handles += [
    Line2D([0], [0], color=trend_color, lw=2),
    Line2D([0], [0], color=new_trend_color, lw=2)
]

labels += [
    r"$L_{\mathrm{FIR}} = 10^{2.99} L'_{\mathrm{HCN}}$",
    rf"$L_\mathrm{{FIR}} = "
    rf"10^{{{log_intercept_med:.2f}\pm{log_intercept_hi-log_intercept_med:.2f}}}"
    rf"L'_\mathrm{{HCN}}^{{{slope_med:.2f}\pm{slope_hi-slope_med:.2f}}}$",
]

lookup = dict(zip(labels, handles))

order = [
    "Parts of Galaxies",
    "z~0",
    "DSFGs",
    "Quasars",
    "Rybak 2026",
    # "Westoby 2026",
    "This work",
    r"$L_{\mathrm{FIR}} = 10^{2.99} L'_{\mathrm{HCN}}$",
    rf"$L_\mathrm{{FIR}} = "
    rf"10^{{{log_intercept_med:.2f}\pm{log_intercept_hi-log_intercept_med:.2f}}}"
    rf"L'_\mathrm{{HCN}}^{{{slope_med:.2f}\pm{slope_hi-slope_med:.2f}}}$",
]

# First legend
legend1 = ax.legend(
    [lookup[l] for l in order[:-2]],
    order[:-2],
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    fontsize=10,
    frameon=False
)
ax.add_artist(legend1)

# Second legend (equation)
legend2 = ax.legend(
    [lookup[order[-2]], lookup[order[-1]]],
    [order[-2], order[-1]],
    loc='lower right',
    #bbox_to_anchor=(0.01, 0.70),
    fontsize=14,
    frameon=False,
    handlelength=1
)


#############################################################################################################################################
plt.gcf().set_size_inches(6,6)

plt.savefig('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/FIR_HCN_result.png', dpi = 200,bbox_inches='tight')