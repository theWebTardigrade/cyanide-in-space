import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


LINE_FREQUENCIES = {
    "CO": 115.271,       # GHz
    "HCO+": 89.1885,     # GHz
    "HCN": 88.6318,      # GHz
}


from load_moldata import LevelCalculator
from cloud_model import (
    solve_escape_probabilities, luminosity_per_volume, rho_dot_star,
)
from config import (
    GALAXY_CASES, LINE_COLORS, LINE_LABELS, MOLDATA_FILES, COLL_PARTNER_IDX, get_cloud_kwargs,
)

SPECIES = ["CO", "HCO+", "HCN"]
MAX_LEVEL = 15  # enough levels for CO/HCO+/HCN 1-0 luminosities up to n~1e6-1e8 cm^-3

PAPER_NCRIT = {
    "CO": 560.0,
    "HCO+": 4.6e4,
    "HCN": 2.8e5,
}


def load_molecules():
    """Load each species' LAMDA file once and reuse across all figures."""
    return {sp: LevelCalculator(MOLDATA_FILES[sp], max_level=MAX_LEVEL) for sp in SPECIES}


def make_figure1(mols, case="intermediate", n_means=(1e2, 1e3, 1e4), outfile="figure1.png"):
    fig, axes = plt.subplots(3, 1, figsize=(6, 11), sharex=True)
    lnx_grid = np.linspace(-8, 14, 300)

    for ax, sp in zip(axes, SPECIES):
        level_calc = mols[sp]
        part = COLL_PARTNER_IDX[sp]
        kwargs = get_cloud_kwargs(sp, case)
        line_colors = ['#4285f4', '#FFAD0AFF','#9900ff']

        for n_mean, col in zip(n_means, line_colors):
            beta, R = solve_escape_probabilities(level_calc, n_ref=n_mean,
                                                   coll_partner_idx=part, **kwargs)
            res = luminosity_per_volume(level_calc, beta, n_mean=n_mean,
                                         coll_partner_idx=part, lnx_grid=lnx_grid,
                                         T=kwargs["T"], mach=kwargs["mach"], X_abund=kwargs["X_abund"])

            n_grid = res["n"]
            Lnorm = res["dLdlnx"] / res["L"] if res["L"] > 0 else res["dLdlnx"] * 0
            Mnorm = res["dMdlnx"] / res["M"]

            exponent = int(np.log10(n_mean))
            ax.plot(n_grid, Lnorm, color=col, lw=2, label=rf"$\bar n=10^{{{exponent}}}$ cm$^{{-3}}$")
            ax.plot(n_grid, Mnorm, color=col, lw=2, ls=":")

            beta10 = beta[1, 0]
            ncrit = beta10 * PAPER_NCRIT[sp]

            ax.axvline(ncrit, color='black', lw=1, ls="--", alpha=0.7)

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_ylim(0, 0.29)
        ax.set_xlim(1e1, 1e9)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="y", which="minor", length=3)
        ax.set_ylabel(r"$L^{-1} \, [dL/d\ln n]$  (solid)" "\n" r"$M^{-1} \, [dM/d\ln n]$  (dotted)", fontsize=12)
        ax.set_title(f"{LINE_LABELS[sp]}", loc="left", fontsize=16,  y=1.0, pad=-20)
        ax.legend(fontsize=12, loc="upper right", frameon=False)

    axes[-1].set_xlabel(r"$n$ (cm$^{-3}$)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print("wrote", outfile)



# FIGURE 2
def make_figure2(mols, outfile="figure2.png"):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(6, 11),
        sharex=True
    )

    n_grid = np.geomspace(10, 1e7, 20)

    LSUN = 3.828e33
    PC_CM = 3.0856776e18  # cm


    SPECIES_COLORS = {
        "CO": "#4285f4",
        "HCO+": "#FFAD0A",
        "HCN": "#9900ff",
    }

    GALAXY_COLORS = {
        "CO": {
            "normal": "#b3cefb",
            "intermediate": "#4285f4",
            "starburst": "#1a3562",
        },
        "HCO+": {
            "normal": "#ffde9d",
            "intermediate": "#FFAD0A",
            "starburst": "#664504",
        },
        "HCN": {
            "normal": "#d699ff",
            "intermediate": "#9900ff",
            "starburst": "#3d0066",
        },
    }


    # --------------------------------------------------
    # Compute SFR/L'
    # --------------------------------------------------

    ratios = {}

    for sp in SPECIES:

        level_calc = mols[sp]
        part = COLL_PARTNER_IDX[sp]

        for case in GALAXY_CASES:

            kwargs = get_cloud_kwargs(sp, case)

            vals = []

            for n_mean in n_grid:

                beta, R = solve_escape_probabilities(
                    level_calc,
                    n_ref=n_mean,
                    coll_partner_idx=part,
                    **kwargs
                )


                res = luminosity_per_volume(
                    level_calc,
                    beta,
                    n_mean=n_mean,
                    coll_partner_idx=part,
                    T=kwargs["T"],
                    mach=kwargs["mach"],
                    X_abund=kwargs["X_abund"]
                )


                # Msun yr^-1 pc^-3
                sfr = rho_dot_star(n_mean,kwargs["mach"])


                # Frequency
                nu = LINE_FREQUENCIES[sp]
                L_solar_density = res["L"] / LSUN

                # L'/V
                Lprime_density = L_solar_density/(3e-11 * nu**3)


                # cm^-1 -> pc^-1
                Lprime_density *= PC_CM**3


                vals.append(
                    sfr / Lprime_density
                )


            ratios[(sp, case)] = np.array(vals)



    # --------------------------------------------------
    # Panel 1: all species
    # --------------------------------------------------

    ax = axes[0]

    for sp in SPECIES:

        ax.plot(
            n_grid,
            ratios[(sp, "intermediate")],
            color=SPECIES_COLORS[sp],
            lw=2,
            label=LINE_LABELS[sp],
        )


    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e1,1e7)
    ax.set_ylim(5e-9,1e-5)


    ax.text(
        0.02,
        0.96,
        "Intermediate galaxy",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )


    ax.legend(
        frameon=False,
        loc = 'lower right'
    )


    # Add FIR/L' secondary axis
    ax2 = ax.twinx()

    ax2.set_yscale("log")


    ax2.set_ylim(
        ax.get_ylim()[0] * 5.8e9,
        ax.get_ylim()[1] * 5.8e9
    )

    ax2.set_ylabel("")



    # --------------------------------------------------
    # Panels 2-4
    # --------------------------------------------------

    for ax, sp in zip(axes[1:], SPECIES):

        for case in GALAXY_CASES:

            ax.plot(
                n_grid,
                ratios[(sp, case)],
                color=GALAXY_COLORS[sp][case],
                lw=2,
                label=case.capitalize(),
                zorder=10
            )


        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(1e1,1e7)
        ax.set_ylim(5e-9,1e-5)


        ax.text(
            0.02,
            0.96,
            LINE_LABELS[sp],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
        )


        ax.legend(
            frameon=False,
            loc="lower right"
        )


        # Right axis: L_FIR/L'
        ax2 = ax.twinx()

        ax2.set_yscale("log")

        ax2.set_ylim(
            ax.get_ylim()[0] * 5.8e9,
            ax.get_ylim()[1] * 5.8e9
        )

        ax2.set_ylabel("")

        # Add Gao & Solomon HCN relation band
        if sp == "HCN":

            # L_FIR/L' central value and scatter
            log_LFIR_Lp = 2.99
            dex = 0.3

            LFIR_Lp_central = 10**log_LFIR_Lp
            LFIR_Lp_low = 10**(log_LFIR_Lp - dex)
            LFIR_Lp_high = 10**(log_LFIR_Lp + dex)

            # Convert to SFR/L' left-axis units
            sfr_Lp_central = LFIR_Lp_central / 5.8e9
            sfr_Lp_low = LFIR_Lp_low / 5.8e9
            sfr_Lp_high = LFIR_Lp_high / 5.8e9

            # Horizontal line
            ax.axhline(
                sfr_Lp_central,
                color="grey",
                lw=1.5,
                ls="--",
                zorder=0,
            )

            # 0.3 dex shaded region
            ax.axhspan(
                sfr_Lp_low,
                sfr_Lp_high,
                color="grey",
                alpha=0.25,
                zorder=0,
            )


    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    for ax in axes[:-1]:

        ax.tick_params(
            labelbottom=False
        )


    for ax in axes:

        ax.minorticks_on()

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=False,
        )


    fig.supylabel(
        r"SFR "
        r"$/\, L' \, M_\odot\,\mathrm{yr}^{-1} \,/\, "
        r"[\mathrm{K\,km\,s^{-1}\,pc^{2}}]$",
        x=0.005
    )

    fig.text(
        1.01,
        0.5,
        r"$L_{\rm FIR} \,/\, L' \qquad "
        r"[L_\odot/(\mathrm{K\,km\,s^{-1}\,pc^2})]$",
        rotation=90,
        va="center",
        ha="right",
    )


    axes[-1].set_xlabel(
        r"$\bar{n}$ (cm$^{-3}$)"
    )


    fig.subplots_adjust(
        hspace=0
    )


    fig.savefig(
        outfile,
        dpi=150,
        bbox_inches="tight",
    )

    print("wrote", outfile)




if __name__ == "__main__":
    mols = load_molecules()
    #make_figure1(mols)
    make_figure2(mols)
