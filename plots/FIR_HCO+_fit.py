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
from scipy.stats import spearmanr
from scipy.stats import ks_2samp

import linmix
from scipy.optimize import curve_fit

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"


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


Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/table_ULIRGS_GC2008_HCO+.data')
GC2008_LFIR, GC2008_LHCO = np.genfromtxt(Table, unpack=True, usecols=(1, 2))
Table.close()
GC2008_LFIR = GC2008_LFIR*1e11
GC2008_LHCO = GC2008_LHCO*1e8

Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/table_Krips2008_HCO+.data')
Krips_LFIR, Krips_LHCO = np.genfromtxt(Table, unpack=True, usecols=(1, 2))
Table.close()
Krips_LFIR = Krips_LFIR*1e10

Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/table_ULIRGS_GB2012_HCO+.data')
GB2012_LFIR, GB2012_LHCO = np.genfromtxt(Table, unpack=True, usecols=(1, 2))
Table.close()
GB2012_LFIR = GB2012_LFIR*1e11


Table = open('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/table_Privon2015_HCO+.data')
Privon_LFIR, Privon_LHCO = np.genfromtxt(Table, unpack=True, usecols=(1, 2))
Table.close()
Privon_LFIR = 10**Privon_LFIR


######################################################################################################################
######################################################################################################################

fig = plt.figure()
ax=fig.add_subplot(111)
plt.xlabel(r"$L'_{\mathrm{HCO}^+(1-0)}$ [K km s$^{-1}$ pc$^2$]", fontsize=16)
plt.ylabel(r'$L_\text{FIR}$ [L$_\odot$]', fontsize=16)


plt.tick_params(axis='both', which = 'major', length=10, direction = 'in', width=0.5, color = 'black', labelsize = 10, top=True, right=True)
plt.tick_params(axis='both', which = 'minor',length=5, direction = 'in', width=0.5, color = 'black', labelsize = 10, top=True, right=True)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e6, 1e12)
plt.ylim(1e9, 1e15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


def UL(x, y, color, label, edgecolor=None, size=None, zorder=5, marker_new=None):
    plt.plot([0.70*x, x], [y, y], c = color, lw=2 , zorder = zorder) 
    plt.scatter([x], [y], facecolor=color, edgecolor = 'black',s=size, zorder=zorder, label= label, marker=marker_new) 
    plt.scatter([0.70*x], [y], c = color, edgecolor = 'black', s=30, marker = '<', zorder = zorder)


color_palette=['#4285f4', '#D72000FF','#2CA030FF', '#EE6100FF', '#FFAD0AFF', '#1BB6AFFF', '#9093A2FF', '#132157FF', '#9900ff']


###########################################################################################################################
plt.scatter(GC2008_LHCO, GC2008_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

plt.scatter(Krips_LHCO, Krips_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

plt.scatter(GB2012_LHCO, GB2012_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)

plt.scatter(Privon_LHCO, Privon_LFIR, color = '#9093A2FF', marker='x', label = 'z~0', zorder=5, s=20)


####################

rybak_color = '#4285f4'
rybak_size = 7
plt.errorbar(85e8/(0.36*8.8), 66e12/8.8, xerr=130e8/8.8, color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar((73e8/(0.36*10.9)), 64e12/10.9, xerr=25e8/(0.36*10.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar((113e8/(0.36*15.3)), 66e12/15.3, xerr=19e8/(0.36*15.3), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar((80e8/(0.27*4.9)), 71e12/4.9, xerr=16e8/(0.27*4.9), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
UL((75e8/(0.27*1.8)), 28e12/1.8, color = rybak_color, label = 'Rybak 2026', size=50, marker_new='s')
plt.errorbar((310e8/(0.36*14.7)), 133e12/14.7, xerr=19e8/(0.36*14.7), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar((61e8/(0.27)), 40e12, xerr=15e8/(0.27), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
UL((140e8/(8.6)), 32e12/8.6, color = rybak_color, label = 'Rybak 2026', size=50, marker_new='s')
plt.errorbar((170e8/(0.27*4.1)), 75e12/4.1, xerr=30e8/(0.27*4.1), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar(160e8/(0.27*18.2), 43e12/18.2, xerr=43e8/(0.27*18.2), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')
plt.errorbar((146e8/(0.27*9.2)), 90e12/9.2, xerr=14e8/(0.27*9.2), color = rybak_color, mec='black', label = 'Rybak 2026', zorder=5, markersize=rybak_size, fmt='s')



beth_color = '#EE6100FF'
beth_size = 15
r21 = 0.7
# plt.errorbar(1.87e13, 5.90e10/(r21*5.5), yerr=6.75e9/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0314-44
# plt.errorbar(4.68e12, 5.09e10/(r21*21), yerr=7.71e9/(r21*21), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2134-50
# plt.errorbar(9.94e12, 5.58e10/(r21*10.04), yerr=8.58e9/(r21*10.04), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0532-50
# plt.errorbar(1.4e13, 6.42e10/(r21*5.5), yerr=1.19e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-58
# plt.errorbar(4.62e13, 7.85e10/(r21*5.5), yerr=7.26e9/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0402-45
# plt.errorbar(4.09e12, 2.94e10/(r21*20.1), yerr=6.19e9/(r21*20.1), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0538-50
# plt.errorbar(1.58e13, 3.04e10/(r21*5.5), yerr=1.14e10/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2101-60
# plt.errorbar(2.34e13, 2.09e10/(r21*5.5), yerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0625-55
# plt.errorbar(2.22e13, 1.04e10/(r21*5.5), yerr=0/(r21*5.5), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT2332-53
# plt.errorbar(1.87e13, 1.35e10/(r21*5.49), yerr=6.58e9/(r21*5.49), color = beth_color, label = 'Westoby 2026', zorder=5, markersize=beth_size, fmt='.') # SPT0027-50 


my_color = '#9900ff'
my_size= 15
UL(5.2e10, 2.2e14, color = my_color, edgecolor='#9900ff', label = 'This work', size=50, zorder=10, marker_new='o')
plt.errorbar(8.7e10/8.3, 1.77e14/8.3, xerr=2.9e10/8.3, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')
plt.errorbar(26.4e10/28.2, 4.60e14/28.2, xerr=4.3e10/28.2, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')
plt.errorbar(6.6e10/7.6, 2.3e14/7.6, xerr=2.0e10/7.6, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')
plt.errorbar(12.1e10/25, 8.110e13/25, xerr=2.3e10/25, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')
plt.errorbar((9.6e10/8), 12e13/8, xerr=(1.9e10/8), color = my_color, mec='black', label = 'This work', zorder=6, markersize=my_size, fmt='.')
UL(4.6e10/32.8, 7.4e13/32.5, color = my_color, edgecolor='#9900ff', label = 'This work', size=50, zorder=10, marker_new='o')
UL(10.4e10/32.8, 10.06e13/11.7, color = my_color, edgecolor='#9900ff', label = 'This work', size=50, zorder=10, marker_new='o')
UL(4.63e10/6.1, 4.11e13/6.1, color = my_color, edgecolor='#9900ff', label = 'This work', size=50, zorder=10, marker_new='o')
plt.errorbar(2.7e10/57, 9.23e13/57, xerr=1.6e10/57, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')
plt.errorbar(19.0e10/11, 3.90e14/11, xerr=2.2e10/11, color = my_color, mec='black', label = 'This work', zorder=10, markersize=my_size, fmt='.')

##############
# trend_color1 = '#D72000FF'
# plt.plot([1.e8, 1.e15], [(10**-2.99)*1e8, (10**-2.99)*1e15], color=trend_color1, lw=1, linestyle = 'dashed')
# 0.37 dex

# plt.fill_between(
#     [1.e8, 1.e15],
#     [(10**-2.99)*10**0.30*1e8, (10**-2.99)*10**0.30*1e15],
#     [(10**-2.99)/10**0.30*1e8, (10**-2.99)/10**0.30*1e15],
#     alpha=0.2,
#     color=trend_color1
# )

# plt.plot([1.e8, 1.e15], [1./977.*10**0.37*1e8, 1./977.*10**0.37*1e15], color=trend_color1, linestyle='dashed')
# plt.plot([1.e8, 1.e15], [1./977./10**0.37*1e8, 1./977./10**0.37*1e15], color=trend_color1, linestyle='dashed')

###########################################################################################################################################

all_LFIR = np.concatenate([
                        np.abs(GC2008_LFIR), 
                        np.abs(Krips_LFIR), 
                        np.abs(GB2012_LFIR), 
                        np.abs(Privon_LFIR), 

                        np.array([
                        66e12/8.8,      # SDP.9
                        64e12/10.9,     # SDP.11
                        66e12/15.3,     # G09v1.40
                        71e12/4.9,      # SDP.17
                        28e12/1.8,      # G15v2.235 (UL)
                        133e12/14.7,    # J0209
                        40e12,          # G09v1.326
                        32e12/8.6,      # SDP.130 (UL)
                        75e12/4.1,      # NAv1.195
                        43e12/18.2,     # SDP.81
                        90e12/9.2,      # G12v2.43

                        2.2e14,          # W2246 (UL)
                        1.77e14/8.3,     # J1336
                        4.60e14/28.2,    # J0226
                        2.3e14/7.6,      # J1053
                        8.110e13/25,     # J1202
                        12e13/8,         # J1323
                        7.4e13/32.5,     # Eyelash (UL)
                        10.06e13/11.7,   # Nav1.56 (UL)
                        4.11e13/6.1,     # J213+0109 (UL)
                        9.23e13/57,      # J0116
                        3.90e14/11      # Eyebrow
                        ])])


all_LHCO = np.concatenate([np.abs(GC2008_LHCO), 
                           np.abs(Krips_LHCO), 
                           np.abs(GB2012_LHCO), 
                           np.abs(Privon_LHCO),

                            np.array([
                            85e8/(0.36*8.8),      # SDP.9
                            73e8/(0.36*10.9),     # SDP.11
                            113e8/(0.36*15.3),    # G09v1.40
                            80e8/(0.27*4.9),      # SDP.17
                            75e8/(0.27*1.8),      # G15v2.235 (UL)
                            310e8/(0.36*14.7),    # J0209
                            61e8/(0.27),    # G09v1.326
                            140e8/8.6,            # SDP.130 (UL)
                            170e8/(0.27*4.1),     # NAv1.195
                            160e8/(0.27*18.2),    # SDP.81
                            146e8/(0.27*9.2),     # G12v2.43

                            5.2e10,          # W2246 (UL)
                            8.7e10/8.3,      # J1336
                            26.4e10/28.2,    # J0226
                            6.6e10/7.6,      # J1053
                            12.1e10/25,      # J1202
                            9.6e10/8,        # J1323
                            4.6e10/32.8,     # Eyelash (UL)
                            10.4e10/32.8,    # Nav1.56 (UL)
                            4.63e10/6.1,     # J213+0109 (UL)
                            2.7e10/57,       # J0116
                            19.0e10/11      # Eyebrow
                        ])])


all_LHCO_errs = np.concatenate([
    np.zeros_like(GC2008_LHCO),
    np.zeros_like(Krips_LHCO),
    np.zeros_like(GB2012_LHCO),
    np.zeros_like(Privon_LHCO),

    np.array([
    130e8/8.8,            # SDP.9
    25e8/(0.36*10.9),     # SDP.11
    19e8/(0.36*15.3),     # G09v1.40
    16e8/(0.27*4.9),      # SDP.17
    0,                 # G15v2.235 (UL)
    19e8/(0.36*14.7),     # J0209
    15e8/(0.27*40e12),    # G09v1.326
    0,                 # SDP.130 (UL)
    30e8/(0.27*4.1),      # NAv1.195
    43e8/(0.27*18.2),     # SDP.81
    14e8/(0.27*9.2),      # G12v2.43

    0,             # W2246 (UL)
    2.9e10/8.3,       # J1336
    4.3e10/28.2,      # J0226
    2.0e10/7.6,       # J1053
    2.3e10/25,        # J1202
    1.9e10/8,         # J1323
    0,             # Eyelash (UL)
    0,             # Nav1.56 (UL)
    0,             # J213+0109 (UL)
    1.6e10/57,        # J0116
    2.2e10/11        # Eyebrow
])])

detections_flags = np.concatenate([
    (GC2008_LHCO > 0).astype(int),
    (Krips_LHCO > 0).astype(int),
    (GB2012_LHCO > 0).astype(int),
    (Privon_LHCO > 0).astype(int),

    np.array([
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,

    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    1
])])



#############################################################################################################################################

log_xvalues = np.log10(all_LFIR)
log_yvalues = np.log10(all_LHCO)
log_yerrs = np.array(all_LHCO_errs) / (np.array(all_LHCO) * np.log(10))


linmix_model = linmix.LinMix(log_xvalues, log_yvalues, ysig=log_yerrs, delta=detections_flags, seed=42) 
linmix_model.run_mcmc(silent=True)

alpha_chain3 = linmix_model.chain['alpha']
beta_chain3 = linmix_model.chain['beta']

alpha_med3 = np.median(alpha_chain3)
alpha_lo3, alpha_hi3 = np.percentile(alpha_chain3, [16, 84])

beta_med3 = np.median(beta_chain3)
beta_lo3, beta_hi3 = np.percentile(beta_chain3, [16, 84])

scatter3 = np.sqrt(np.median(linmix_model.chain['sigsqr']))

intercept3 = alpha_med3
slope3 = beta_med3

print(f"Best-fit slope: {slope3:.2f}")
print(f"Best-fit intercept: {intercept3:.2f}")
print(f"Scatter: {scatter3:.2f}")

intercept4_chain = -alpha_chain3 / beta_chain3
slope4_chain = 1. / beta_chain3

intercept4 = np.median(intercept4_chain)
intercept4_lo, intercept4_hi = np.percentile(intercept4_chain, [16, 84])

slope4 = np.median(slope4_chain)
slope4_lo, slope4_hi = np.percentile(slope4_chain, [16, 84])


x_line = np.logspace(9, 15, 300)

log_y_line = intercept3 + slope3 * np.log10(x_line)

y_line = 10**log_y_line

trend_color3 = '#FFAD0AFF'

plt.plot(
    y_line,
    x_line,
    color=trend_color3,
    lw=1.5,
    linestyle='-'
)

plt.fill_betweenx(
    x_line,
    10**(log_y_line - scatter3),
    10**(log_y_line + scatter3),
    alpha=0.2,
    color=trend_color3,
    edgecolor = None
)

residuals = log_yvalues - (intercept3 + slope3 * log_xvalues)

scatter_first92 = np.std(residuals[:92], ddof=1)

scatter_rest = np.std(residuals[92:], ddof=1)

print(f"Scatter (z=0): {scatter_first92:.3f} dex")
print(f"Scatter (high z): {scatter_rest:.3f} dex")

#############################################################################################################################################
handles, labels = ax.get_legend_handles_labels()

handles += [
    Line2D([0], [0], color=trend_color3, lw=3)
]

labels += [
    rf"$L_\mathrm{{FIR}} = "
    rf"10^{{{intercept4:.2f}\pm{(intercept4 - intercept4_lo):.2f}}}"
    rf"L'^{{\,{slope4:.2f}\pm{(slope4 - slope4_lo):.2f}}}"
    rf"_\mathrm{{HCO^+}}$"
]

lookup = dict(zip(labels, handles))

order = [
    "z~0",
    "Rybak 2026",
    "This work",

    rf"$L_\mathrm{{FIR}} = "
    rf"10^{{{intercept4:.2f}\pm{(intercept4 - intercept4_lo):.2f}}}"
    rf"L'^{{\,{slope4:.2f}\pm{(slope4 - slope4_lo):.2f}}}"
    rf"_\mathrm{{HCO^+}}$"
]

legend1 = ax.legend(
    [lookup[l] for l in order[:-1]],
    order[:-1],
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    fontsize=10,
    frameon=False
)
ax.add_artist(legend1)

legend2 = ax.legend(
    [lookup[order[-1]]],
    [order[-1]],
    loc='lower right', 
    fontsize=14,
    frameon=False,
    handlelength=1
)

#############################################################################################################################################
plt.gcf().set_size_inches(6,6)

plt.savefig('/home/polaris/cyanide_in_space/scripts/prussic-main/prussic_iii/literature_data/FIR_HCO+.png', dpi = 200,bbox_inches='tight')