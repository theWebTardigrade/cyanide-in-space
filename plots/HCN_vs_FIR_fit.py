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
Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/table_Gao2004.data')
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

#### Grácia-Carpio 2008 ??


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
plt.xlabel(r'$L_\text{FIR}$ [L$_\odot$]', fontsize=16)
plt.ylabel(r"$L'_{\mathrm{HCN(1-0)}}$ [K km s$^{-1}$ pc$^2$]", fontsize=16)


plt.tick_params(axis='both', which = 'major', length=10, direction = 'in', width=0.5, color = 'black', labelsize = 13, top=True, right=True)
plt.tick_params(axis='both', which = 'minor',length=5, direction = 'in', width=0.5, color = 'black', labelsize = 13, top=True, right=True)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e9, 1e15)
plt.ylim(1e6,1e12)

def UL(x, y, color, label, edgecolor=None, size=None, zorder=5, marker_new=None):
    plt.plot([x,x], [0.70*y, y], c = edgecolor, lw=2 , zorder = zorder) 
    plt.scatter([x], [y], facecolor=color, edgecolor = 'black',s=size, zorder=zorder, label= label, marker=marker_new) 
    plt.scatter([x], [0.70*y], c = color, edgecolor = 'black', s=30, marker = 'v', zorder = zorder)


# Color Pallete
color_palette=['#4285f4', '#D72000FF','#2CA030FF', '#EE6100FF', '#FFAD0AFF', '#1BB6AFFF', '#9093A2FF', '#132157FF', '#9900ff']


###########################################################################################################################
#### GAO & SOLOMON 2004 #### z=0
plt.scatter(GS2004_LFIR, GS2004_LHCN, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Krips 2008 #### z=0
plt.scatter(10.0**Krips_LFIR, 10.0**Krips_LHCN, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Garcia-Burillo 2012 #### z=0
plt.scatter(GB2012_LFIR, GB2012_LHCN, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

### Privon 2015 #### z=0
plt.scatter(P2015_LFIR, P2015_LHCN, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)
###########################################################################################################################



#### GAO 2007 #### DSFGs
UL(0.93e12, 0.6e9, label = 'DSFGs', color='black', edgecolor='black', size=40) # J16359 6612(B)
UL(22e12, 28e9, label = 'DSFGs', color='black', edgecolor='black', size=40) #J04135+10277
UL(28e12, 46e9, label = 'DSFGs', color='black', edgecolor='black', size=40) # SMM J02399 0136
UL(6.1e12, 3.7e9, label = 'DSFGs', color='black', edgecolor='black', size=40) #SMM J02396+0134
UL(1.5e12, 1.6e9, label = 'DSFGs', color='black', edgecolor='black', size=40) # J14011+0252; Carilli+2005


# NCv1.143 DSFG, (4-3) 

# QUASARS ####
# Upper limits
UL(55e12, 39e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #BR 1202 0725
UL(20e12, 9.3e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #SDSS J1148+5251
UL(2.1e12, 0.6e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #J0911+0551
UL(2.7e12, 0.9e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, size=40) #MG0751 2716

# Detections
plt.scatter(6.2e12, 3.2e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # Cloverleaf
plt.scatter(3.4e12, 1.2e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # IRAS F10214+4724
plt.scatter(17e12, 6.7e9, label = 'Quasars', color='white', edgecolor='black', zorder=3, s=40, marker='o') # VCV J1409+5628

# APM 08279+5255 Quasar, (4-3) 




### RYBAK 2026
rybak_color = '#4285f4'
rybak_size = 7
plt.errorbar(66e12/8.8, 380e8/8.8, yerr=130e8/8.8, color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.9
plt.errorbar(64e12/10.9, (170e8/(0.59*10.9)), yerr=25e8/(0.59*10.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.11
plt.errorbar(66e12/15.3, (94e8/(0.59*15.3)), yerr=19e8/(0.59*15.3), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')#G09v1.40
plt.errorbar(71e12/4.9, (126e8/(0.41*4.9)), yerr=16e8/(0.41*4.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.17
UL(28e12/1.8, (74e8/(0.41*1.8)), color = rybak_color, label = 'Rybak 2026', size=40, marker_new='s') #G15v2.235
plt.errorbar(133e12/14.7, (560e8/(0.59*14.7)), yerr=19e8/(0.59*14.7), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #J0209
plt.errorbar(40e12,  (52e8/0.41), yerr=13e8/(0.41),color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #G09v1.326
plt.errorbar(32e12/8.6, (52e8/(0.41*8.6)), yerr=23e8/(0.41*8.6), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #SDP.130
plt.errorbar(75e12/4.1, (180e8/(0.41*4.1)), yerr=30e8/(0.41*4.1), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #NAv1.195
UL(43e12/18.2, 3.7e10/18.2, label = 'Rybak 2026', color=rybak_color, edgecolor=rybak_color, size=40, marker_new='s') #SDP.81
plt.errorbar(90e12/9.2, (210e8/(0.41*9.2)),yerr=15e8/(0.41*9.2), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s') #G12v2.43

### 2022
UL(36e12/5.3, 5.1e10/5.3, color = rybak_color, label = 'Rybak 2026', size=35, marker_new='s') #HXMM.02
UL(156e12/15.4, 11.4e10/15.4, color = rybak_color, label = 'Rybak 2026', size=35, marker_new='s') #J1609



### BETH WESTOBY 2026
beth_color = '#EE6100FF'
beth_size = 15
r21 = 0.7
plt.errorbar(1.87e13, 5.90e10/(r21*5.5), yerr=6.75e9/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0314-44
plt.errorbar(4.68e12, 5.09e10/(r21*21), yerr=7.71e9/(r21*21), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2134-50
plt.errorbar(9.94e12, 5.58e10/(r21*10.04), yerr=8.58e9/(r21*10.04), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0532-50
plt.errorbar(1.4e13, 6.42e10/(r21*5.5), yerr=1.19e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-58
plt.errorbar(4.62e13, 7.85e10/(r21*5.5), yerr=7.26e9/(r21 *5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0402-45
plt.errorbar(4.09e12, 2.94e10/(r21*20.1), yerr=6.19e9/(r21*20.1), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0538-50
plt.errorbar(1.58e13, 3.04e10/(r21*5.5), yerr=1.14e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2101-60
plt.errorbar(2.34e13, 2.09e10/(r21*5.5), yerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-55
plt.errorbar(2.22e13, 1.04e10/(r21*5.5), yerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2332-53
plt.errorbar(1.87e13, 1.35e10/(r21*5.49), yerr=6.58e9/(r21*5.49), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0027-50 


# MPF 2026
my_color = "#9151B8"
my_edge_color = '#000000'
my_size= 15
###########
#UL(10**(13.73), 5.5e10, color = my_color, edgecolor='#9900ff', label = 'This work', size=50, zorder=10) # W2246
UL(2.2e14, 5.5e10, color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # W2246
###########

plt.errorbar(1.77e14/8.3, 11.7e10/8.3, yerr=3.2e10/8.3, color = my_color, mec=my_edge_color ,label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1336
plt.errorbar(4.60e14/28.2, 29.1e10/28.2, yerr=4.2e10/28.2, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J0226
plt.errorbar(2.3e14/7.6, 13e10/7.6, yerr=2.1e10/7.6, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1053
plt.errorbar(8.110e13/25, 14.1e10/25, yerr=2.3e10/25, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J1202
plt.errorbar(12e13/8, (5.5e10/8), yerr=(1.7e10/8), color = my_color, mec=my_edge_color, label = 'This work', zorder=6, markersize=my_size, fmt='.') # J1323
UL(7.4e13/32.5, 4.4e10/32.8, color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # Eyelash
plt.errorbar(10.06e13/11.7, 5.3e10/11.7, yerr=3.3e10/11.7,  color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # Nav1.56
UL(4.11e13/6.1, 4.38e10/6.1, color = my_color, edgecolor=my_color, label = 'This work', size=50, zorder=10) # J213+0109
plt.errorbar(9.23e13/7.0, 3.8e10/7.0, yerr=1.0e10/7.0, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # J0116

# Wrong
plt.errorbar(39e13/11, 4.2e10/11, yerr=2.2e10/11, color = my_color, mec=my_edge_color, label = 'This work', zorder=10, markersize=my_size, fmt='.') # Eyebrow


########################################################################################################################
# Jimenez-Donaire 2019 trend
trend_color = '#D72000FF'

plt.plot([1.e8, 1.e15], [(10**-2.99)*1e8, (10**-2.99)*1e15], color=trend_color, lw=1, linestyle = '-')


plt.fill_between(
    [1.e8, 1.e15],
    [(10**-2.99)*10**0.30*1e8, (10**-2.99)*10**0.30*1e15],
    [(10**-2.99)/10**0.30*1e8, (10**-2.99)/10**0.30*1e15],
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


plt.scatter(10**usero2015['logLIR'], 10**usero2015['logLHCN'], color = '#9093A2FF', marker='.', label = 'Parts of Galaxies', zorder=5, s=20)

###########################################################################################################################################


#####
# FOR FITTING
# Coment what you don't want to include
#####

x_values = [
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
    #2.2e14,
    1.77e14/8.3,
    4.60e14/28.2,
    2.3e14/7.6,
    8.110e13/25,
    7.4e13/32.5,
    10.06e13/11.7,
    4.11e13/6.1,
    9.23e13/7.0,
    39e13/11
]

y_values = [
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

    #5.5e10,
    11.7e10/8.3,
    29.1e10/28.2,
    13.0e10/7.6,
    14.1e10/25,
    4.4e10/32.5,
    5.3e10/11.7,
    4.38e10/6.1,
    3.8e10/7.0,
    4.2e10/11
]


y_errs = [
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

    #0,
    3.2e10/8.3,
    4.2e10/28.2,
    2.1e10/7.6,
    2.3e10/25,
    0,
    3.3e10/11.7,
    0,
    1.0e10/7.0,
    2.2e10/11,


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

    #0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
]


################################################################################
# Define what log values to use for the fits


# Rybak2026, Westoby2026 and this work
# Already defined above



# Rybak2026, Westoby2026 and this work + z~0
all_x = np.concatenate([
    GS2004_LFIR,
    10.0**Krips_LFIR,
    GB2012_LFIR,
    P2015_LFIR,

    x_values
])

all_y = np.concatenate([
    np.abs(GS2004_LHCN),
    np.abs(10.0**Krips_LHCN),
    np.abs(GB2012_LHCN),
    np.abs(P2015_LHCN),

    y_values
])

all_errs_y = np.concatenate([
    np.zeros_like(GS2004_LHCN),
    np.zeros_like(10.0**Krips_LHCN),
    np.zeros_like(GB2012_LHCN),
    np.zeros_like(P2015_LHCN),

    y_errs
])


# 1 detection, 0 non-detections
all_detections = np.concatenate([
    np.ones_like(GS2004_LHCN),
    np.ones_like(10.0**Krips_LHCN),
    np.ones_like(GB2012_LHCN),
    np.ones_like(P2015_LHCN),

    detections

])


############################################################
#### Filter for detections
FIR_values = np.asarray(x_values)
HCN_values = np.asarray(y_values)
HCN_errs = np.asarray(y_errs)
detections = np.asarray(detections)

mask = detections == 1

FIR_detections = FIR_values[mask]
HCN_detections = HCN_values[mask]
HCN_errs_detections = HCN_errs[mask]
detections_detections = detections[mask]

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

# If forced at 1e13, then it does not select any literature
lit_mask_highLIR = lit_FIR_all > 10**(12.5)

lit_FIR_highLIR = lit_FIR_all[lit_mask_highLIR]
lit_HCN_highLIR = lit_HCN_all[lit_mask_highLIR]
lit_errs_highLIR = np.zeros_like(lit_HCN_highLIR)
lit_detections_highLIR = np.ones_like(lit_HCN_highLIR)

print(f'Including {lit_mask_highLIR.sum()} literature (z~0) points in all fits fit')

FIR_everything = np.concatenate([FIR_detections, lit_FIR_highLIR])
HCN_everything = np.concatenate([HCN_detections, lit_HCN_highLIR])
HCN_err_everything = np.concatenate([HCN_errs_detections, lit_errs_highLIR])
detections_everything = np.concatenate([detections_detections, lit_detections_highLIR])

###########################################################

x_values = np.asarray(FIR_everything)
y_values = np.asarray(HCN_everything)
y_errs = np.asarray(HCN_err_everything)

log_xvalues = np.log10(x_values)
log_yvalues = np.log10(y_values)

n = len(x_values)

# Convert linear uncertainties to log10 uncertainties
log_y_errs = y_errs / (y_values * np.log(10))

#############################################################
# What is the scatter of Rybak and Pólvora Fonseca points in comparison with JD?

JD_scatter = 0.30


# JD relation
model = log_xvalues - 2.99

# Residuals in log space
log_residuals = log_yvalues - model

# Measured scatter
scatter_fromJD = np.std(log_residuals)

# Total uncertainty
sigma_tot = np.sqrt(log_y_errs**2 + JD_scatter**2)

# Chi-square
chi2_jd = np.sum((log_residuals / sigma_tot)**2)

# Statistics
dof_jd = n - 2
redchi2_jd = chi2_jd / dof_jd

# BIC 
k = 2
bic_jd = chi2_jd + k * np.log(n)

print("Comparing MR2026 and MPF2026 points to JD/GS")
print(f"Scatter = {scatter_fromJD:.3f} dex")
print(f"χ² = {chi2_jd:.2f}")
print(f"Reduced χ² = {redchi2_jd:.2f}")
print(f"BIC = {bic_jd:.2f}")


############################################################
# # FIT LINEAR RELATION 
from scipy.optimize import curve_fit


# ============================
# Fixed slope = 1
# ============================

def constant_model(x, intercept):
    return x + intercept


popt, pcov = curve_fit(
    constant_model,
    log_xvalues,
    log_yvalues,
    sigma=log_y_errs
)

intercept_fixed = popt[0]
intercept_fixed_err = np.sqrt(pcov[0, 0])

a_fixed = 10**intercept_fixed

model_fixed = constant_model(log_xvalues, intercept_fixed)
residuals_fixed = log_yvalues - model_fixed

# Intrinsic scatter (dex)
scatter_fixed = np.std(residuals_fixed)

# Total uncertainty
sigma_tot = np.sqrt(log_y_errs**2 + scatter_fixed**2)

# Chi-square
chi2_fixed = np.sum((residuals_fixed / sigma_tot)**2)

dof_fixed = n - 1
redchi2_fixed = chi2_fixed / dof_fixed

bic_fixed = chi2_fixed + np.log(n)

print("\nFixed slope model:")
print(f"HCN = ({a_fixed:.3e}) FIR")
print(f"Intercept = {intercept_fixed:.3f} ± {intercept_fixed_err:.3f}")
print(f"Scatter = {scatter_fixed:.3f}")
print(f"χ² = {chi2_fixed:.2f}")
print(f"Reduced χ² = {redchi2_fixed:.2f}")
print(f"BIC = {bic_fixed:.2f}")

print(residuals_fixed)


# Plot fixed slope
x_line = np.logspace(
    np.log10(np.min(FIR_detections)),
    np.log10(np.max(FIR_detections)),
    200
)

y_line = a_fixed*x_line

plt.plot(
    x_line,
    y_line,
    color='yellow',
    lw=1,
    label='Slope = 1'
)

plt.fill_between(
    x_line,
    y_line*10**(-scatter_fixed),
    y_line*10**(scatter_fixed),
    color='yellow',
    alpha=0.2
)



# ============================
# Free slope
# ============================

def linear_model(x, slope, intercept):
    return slope*x + intercept


popt, pcov = curve_fit(
    linear_model,
    log_xvalues,
    log_yvalues,
    sigma=log_y_errs
)

slope = popt[0]
intercept = popt[1]

slope_err = np.sqrt(pcov[0, 0])
intercept_err = np.sqrt(pcov[1, 1])

a_free = 10**intercept

model_free = linear_model(
    log_xvalues,
    slope,
    intercept
)

residuals_free = log_yvalues - model_free

# Intrinsic scatter (dex)
scatter_free = np.std(residuals_free)

# Total uncertainty
sigma_tot = np.sqrt(log_y_errs**2+ scatter_free**2)

# Chi-square
chi2_free = np.sum((residuals_free / sigma_tot)**2)

dof_free = n - 2
redchi2_free = chi2_free / dof_free

bic_free = chi2_free + 2 * np.log(n)

print("\nFree slope model:")
print(f"HCN = ({a_free:.3e}) FIR^{slope:.3f}")
print(f"Slope = {slope:.3f} ± {slope_err:.3f}")
print(f"Intercept = {intercept:.3f} ± {intercept_err:.3f}")
print(f"Scatter = {scatter_free:.3f}")
print(f"Residuals = {residuals_free}")
print(f"χ² = {chi2_free:.2f}")
print(f"Reduced χ² = {redchi2_free:.2f}")
print(f"BIC = {bic_free:.2f}")

print("RMS scatter:", np.std(residuals_free))
print("Median residual:", np.median(residuals_free))
print("Mean residual:", np.mean(residuals_free))
print("Typical uncertainty:", np.median(log_y_errs))
print("Reduced chi2:", redchi2_free)


# Plot free slope

y_line = a_free*x_line**slope

plt.plot(
    x_line,
    y_line,
    color='blue',
    lw=1,
    label='Free slope'
)

plt.fill_between(
    x_line,
    y_line*10**(-scatter_free),
    y_line*10**(scatter_free),
    color='blue',
    alpha=0.2
)




# ######################################################################
#  LINMIX

# Currently it is set up for only fitting mine + rybak2026 points
# To fit everything with a single line, we need to substitute the values to all



log_xvalues = np.log10(FIR_detections)
log_yvalues = np.log10(HCN_detections)

log_yerrs = np.array(HCN_errs_detections) / (np.array(HCN_detections) * np.log(10))


# median_FIR = np.median(log_xvalues)
# median_HCN = np.median(log_yvalues)

# log_xvalues = log_xvalues - [median_FIR for i in range(len(log_xvalues))]  # Centering the x-values for better numerical stability
# log_yvalues = log_yvalues - [median_HCN for i in range(len(log_yvalues))]  # Centering the y-values for better numerical stability

linmix_model = linmix.LinMix(log_xvalues, log_yvalues, ysig=log_yerrs, delta=detections_detections, seed=42, nchains=20, parallelize=True)

linmix_model.run_mcmc(silent=True)

# Extract posterior medians 
alpha_chain1 = linmix_model.chain['alpha']
beta_chain1 = linmix_model.chain['beta']

alpha_med1 = np.median(alpha_chain1)
alpha_lo1, alpha_hi1 = np.percentile(alpha_chain1, [16, 84])

beta_med1 = np.median(beta_chain1)
beta_lo1, beta_hi1 = np.percentile(beta_chain1, [16, 84])

beta_err1 = ((beta_hi1 - beta_med1) + (beta_med1 - beta_lo1)) / 2
alpha_err1 = ((alpha_hi1 - alpha_med1) + (alpha_med1 - alpha_lo1)) / 2

scatter1 = np.sqrt(np.median(linmix_model.chain['sigsqr']))

intercept1 = alpha_med1
slope1 = beta_med1

# Best-fit trend line
x_line = np.logspace(11.69, 15, 200)
log_y_line = alpha_med1 + beta_med1 * np.log10(x_line)
y_line = 10**log_y_line

new_trend_color = '#2CA030FF'

plt.plot(
    x_line,
    y_line,
    color=new_trend_color,
    lw=1,
    linestyle='-'
)

# Intrinsic scatter region
plt.fill_between(
    x_line,
    10**(log_y_line - scatter1),
    10**(log_y_line + scatter1),
    alpha=0.2,
    color=new_trend_color,
    edgecolor='none'
)




# Predicted log(HCN)
log_y_model = intercept1 + slope1 * log_xvalues

# Residuals
residuals = log_yvalues - log_y_model


# Include intrinsic scatter from LINMIX
sigma_total = np.sqrt(log_yerrs**2 + scatter1**2)

chi2_linmix = np.sum((residuals / sigma_total)**2)

dof = len(log_yvalues) - 2
chi2_red = chi2_linmix / dof

k_linmix = 2
bic_linmix = chi2_linmix + k_linmix * np.log(n)

print(f"\nLINMIX:")
print(f"HCN = ({10**intercept1:.3e}) FIR^{slope1:.3f}")
print(f"Slope = {slope1:.3f} (+{beta_hi1-beta_med1:.3f}/-{beta_med1-beta_lo1:.3f})")
print(f"Intercept = {intercept1:.3f} (+{alpha_hi1-alpha_med1:.3f}/-{alpha_med1-alpha_lo1:.3f})")
print(f"Scatter = {scatter1:.3f}")
print(f"Residuals = {residuals}")
print(f"  χ²  = {chi2_linmix:.2f}")
print(f"Reduced χ² = {chi2_red:.2f}")
print(f"  BIC = {bic_linmix:.2f}")



##############################################################################


#############################################################################################################################################
handles, labels = ax.get_legend_handles_labels()

handles += [
    Line2D([0], [0], color=trend_color, lw=2),
    Line2D([0], [0], color=new_trend_color, lw=2)
]

labels += [
    r"$L'_{\mathrm{HCN}} = 10^{-2.99} L_{\mathrm{FIR}}$",
    rf"$L'_\mathrm{{HCN}} = "
    rf"10^{{{intercept1:.2f}\pm{alpha_err1:.2f}}}"
    rf"L_\mathrm{{FIR}}^{{{slope1:.2f}\pm{beta_err1:.2f}}}$",
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
    r"$L'_{\mathrm{HCN}} = 10^{-2.99} L_{\mathrm{FIR}}$",
    rf"$L'_\mathrm{{HCN}} = "
    rf"10^{{{intercept1:.2f}\pm{alpha_err1:.2f}}}"
    rf"L_\mathrm{{FIR}}^{{{slope1:.2f}\pm{beta_err1:.2f}}}$",
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

plt.savefig('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/HCN_vs_FIR_fit2.png', dpi = 200,bbox_inches='tight')
