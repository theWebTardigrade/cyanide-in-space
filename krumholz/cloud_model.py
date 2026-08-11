import numpy as np
from utils import coeffMatrix, ordinateVector, findPopulations, collisionCoeffsFromTable


# Define constants

MU_H2 = 3.9e-24        # g, mean mass per H2 molecule (sec. 2.1 of the paper)
G_CGS = 6.67430e-8     # cm^3 g^-1 s^-2
SFR_FF0 = 0.014        # Krumholz & McKee (2005) normalization at M=100, vir=1.3
H_CGS = 6.62607015e-27  # erg s
C_CGS = 2.99792458e10   # cm / s
KB_CGS = 1.380649e-16   # erg / K
AMU_CGS = 1.66053906660e-24  # g
MSUN = 1.98847e33       # g
YR = 3.15576e7          # s
PC = 3.0856776e18       # cm


# ---------------------------------------------------------------------------
# eqs. (1)-(2): cloud density PDF and star formation rate.
# ---------------------------------------------------------------------------

def sigma2_of_mach(mach):
    """Log-density variance, sigma^2 = ln(1 + 3 M^2 / 4)  (eq. below eq. 1)."""
    return np.log(1.0 + 0.75 * mach ** 2)


def lognormal_dpdlnx(lnx, sigma2):
    """dp/d ln x for the lognormal PDF, eq. (1), with <ln x> = -sigma^2/2."""
    lnxbar = -0.5 * sigma2
    return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(lnx - lnxbar) ** 2 / (2 * sigma2))


def sfr_ff(mach):
    """Star-formation efficiency per free-fall time (Krumholz & McKee 2005)."""
    return SFR_FF0 * (mach / 100.0) ** (-0.32)


def rho_dot_star(n_H2, mach):
    """
    Volumetric star formation rate.

    Input:
        n_H2 : cm^-3
        mach : turbulent Mach number

    Returns:
        M_sun yr^-1 pc^-3
    """

    rho = MU_H2 * n_H2  # g cm^-3

    sfr_cgs = sfr_ff(mach) * np.sqrt(
        32 * G_CGS * rho**3 / (3 * np.pi)
    )
    # g s^-1 cm^-3
    sfr_msun_yr_pc3 = (sfr_cgs * YR * PC**3 / MSUN )

    return sfr_msun_yr_pc3


def sound_speed(T, mu_mean=2.33):
    """Isothermal sound speed of the bulk (mostly H2) gas, cm/s. mu_mean in amu."""
    return np.sqrt(KB_CGS * T / (mu_mean * AMU_CGS))


# ---------------------------------------------------------------------------
# thin wrappers around your LevelCalculator / utils.py machinery
# ---------------------------------------------------------------------------

def list_collision_partners(level_calc):
    """
    Print and return the [(index, name), ...] of collision partners in a
    LevelCalculator, so you can pick the right coll_partner_idx for a given
    molecule.
    """
    partners = []
    for idx, table in enumerate(level_calc.coll_tables):
        name = table.meta.get("name", f"partner {idx}")
        partners.append((idx, name))
        print(f"  [{idx}] {name}")
    return partners


def get_available_temps(level_calc, coll_partner_idx):
    """Sorted array of the tabulated collision temperatures (K) available
    for one collision partner table."""
    table = level_calc.coll_tables[coll_partner_idx]
    temps = []
    for name in table.colnames:
        try:
            temps.append(float(name))
        except ValueError:
            continue
    return np.sort(np.array(temps))


def get_collision_matrix(level_calc, T, coll_partner_idx):
    """
    Get the (n_levels x n_levels) collisional rate coefficient matrix at
    temperature T for one collision partner. Calls utils.collisionCoeffsFromTable
    directly on just that partner's table. LAMDA files only tabulate
    rates at a fixed grid of temperatures, so if T isn't one of them we
    linearly interpolate the rate matrix between the two bracketing
    tabulated temperatures (clipping to the table edge if T is outside
    its range).
    """
    temps = get_available_temps(level_calc, coll_partner_idx)
    table = level_calc.coll_tables[coll_partner_idx]
 
    if np.any(np.isclose(temps, T)):
        T_match = temps[np.argmin(np.abs(temps - T))]
        return collisionCoeffsFromTable(table, T_match, level_calc.table_levels, level_calc.max_level)
 
    T_clip = np.clip(T, temps.min(), temps.max())
    if T_clip != T:
        print(f"warning: T={T}K is outside the tabulated range "
              f"[{temps.min()}, {temps.max()}]K for this collision partner; "
              f"clipping to {T_clip}K")
    lo = temps[temps <= T_clip].max()
    hi = temps[temps >= T_clip].min()
    if lo == hi:
        return collisionCoeffsFromTable(table, lo, level_calc.table_levels, level_calc.max_level)
 
    k_lo = collisionCoeffsFromTable(table, lo, level_calc.table_levels, level_calc.max_level)
    k_hi = collisionCoeffsFromTable(table, hi, level_calc.table_levels, level_calc.max_level)
    w = (T_clip - lo) / (hi - lo)
    return (1 - w) * k_lo + w * k_hi



def level_populations(A_eff, coll_matrix, n_H2):
    """
    Solve the statistical-equilibrium linear system (eqs 5-6) for a single
    density n_H2, given an effective (beta-weighted) Einstein-A matrix,
    using utils.findPopulations exactly as LevelCalculator.find_populations
    does. Returns level populations f_i normalized to sum to 1 (i.e.
    fractional populations, not cm^-3 densities): we get this "for free"
    from findPopulations by asking for a total density of 1 cm^-3
    (dens_partner * abund_to_partner = 1).
    """
    if n_H2 <= 0:
        n_levels = A_eff.shape[0]
        f = np.zeros(n_levels)
        f[0] = 1.0
        return f
    return findPopulations(A_eff, coll_matrix, dens_partner=n_H2, abund_to_partner=1.0 / n_H2)


def transition_info(level_calc, up_idx, low_idx):
    """
    Radiative data for the up_idx -> low_idx transition.
    """
    A_ul = level_calc.A_coeffs[up_idx, low_idx]
    g_u = level_calc.weights[up_idx]
    g_l = level_calc.weights[low_idx]
    dE = level_calc.energies[up_idx] - level_calc.energies[low_idx]  # cm^-1, astropy Quantity
    lam = (1.0 / dE).to("cm").value  # cm
    return A_ul, g_u, g_l, lam


# ---------------------------------------------------------------------------
# eqs. (5)-(8): escape-probability statistical equilibrium
# ---------------------------------------------------------------------------

def solve_escape_probabilities(level_calc, T, mach, X_abund, tau_ref, n_ref,
                                up_idx=1, low_idx=0, coll_partner_idx=1,
                                tol=1e-6, max_iter=300, damping=0.5):
    """
    Self-consistently solve eqs. (5)-(8) for the escape probabilities
    beta_ij and cloud radius R, evaluated at the reference (mean) density
    n_ref, holding tau_ref fixed at a chosen reference transition
    (up_idx -> low_idx, e.g. 1->0), per Table 1 of the paper. Fixed-point
    iteration with damping in place of full Newton-Raphson.

    Returns
    -------
    beta : (n_levels, n_levels) ndarray, beta[j, i] for every radiative
           transition j (upper) -> i (lower) with A_coeffs[j, i] > 0.
           Entries with no radiative transition are left at 1 (unused).
    """

    A_coeffs = level_calc.A_coeffs
    n_levels = A_coeffs.shape[0]
    g = level_calc.weights
    energies = level_calc.energies 
    cs = sound_speed(T)
    coll_matrix = get_collision_matrix(level_calc, T, coll_partner_idx)

    # all radiative transitions present in the data (upper j, lower i)
    pairs = [(j, i) for j in range(n_levels) for i in range(n_levels) if A_coeffs[j, i] > 0]

    beta = np.ones((n_levels, n_levels))

    for _ in range(max_iter):
        A_eff = beta * A_coeffs
        f = level_populations(A_eff, coll_matrix, n_ref)

        # solve for R from the reference transition, holding tau_ref fixed (eq. 8)
        Aij, g_u, g_l, lam = transition_info(level_calc, up_idx, low_idx)
        fj, fi = f[up_idx], f[low_idx]  # fj = upper pop., fi = lower pop.
        geom = (g_u / g_l) * Aij * lam ** 3 / (4 * (2 * np.pi) ** 1.5 * mach * cs)
        denom = geom * n_ref * X_abund * fi * (1 - fj * g_l / (fi * g_u))
        R = tau_ref / denom

        # update tau_ij, beta_ij for every radiative transition (eqs 7-8)
        new_beta = np.ones((n_levels, n_levels))
        for j, i in pairs:
            Aij_, g_u_, g_l_ = A_coeffs[j, i], g[j], g[i]
            dE = energies[j] - energies[i]
            lam_ = (1.0 / dE).to("cm").value
            fj_, fi_ = f[j], f[i]
            if fj_ <= 0 or fi_ <= 0:
                tau = 0.0
            else:
                geom_ = (g_u_ / g_l_) * Aij_ * lam_ ** 3 / (4 * (2 * np.pi) ** 1.5 * mach * cs)
                tau = geom_ * n_ref * X_abund * fi_ * R * (1 - fj_ * g_l_ / (fi_ * g_u_))
                tau = max(tau, 0.0)
            new_beta[j, i] = 1.0 / (1.0 + 0.5 * tau)

        delta = np.max(np.abs(new_beta - beta))
        beta = beta + damping * (new_beta - beta)
        if delta < tol:
            break

    return beta


# ---------------------------------------------------------------------------
# eq. (9): line luminosity per unit volume, integrated over the density PDF
# ---------------------------------------------------------------------------

def level_fraction_vs_density(level_calc, beta, T, coll_partner_idx, n_grid, level_idx):
    """
    With beta held fixed (from solve_escape_probabilities), compute the
    population fraction f_{level_idx}(n) for an array of densities n_grid,
    by re-solving the statistical equilibrium linear system (eqs 5-6) at
    each density with beta_ij fixed (per the paragraph following eq. 9).
    """
    A_eff = beta * level_calc.A_coeffs
    coll_matrix = get_collision_matrix(level_calc, T, coll_partner_idx)
    f_level = np.empty(len(n_grid), dtype=float)
    for k, n_H2 in enumerate(n_grid):
        f = level_populations(A_eff, coll_matrix, n_H2)
        f_level[k] = f[level_idx]
    return f_level


def luminosity_per_volume(level_calc, beta, T, mach, X_abund, n_mean,
                           up_idx=1, low_idx=0, coll_partner_idx=1, lnx_grid=None):
    """
    eq. (9): line luminosity per unit volume in the up_idx -> low_idx
    transition,
        L = X(S) * beta_ul * A_ul * h * nu * Int f_upper(n) n dp/dlnx dlnx
    Also returns dL/dlnx and dM/dlnx (unnormalized integrands) for
    Figure-1-style plots.
    """
    sigma2 = sigma2_of_mach(mach)
    if lnx_grid is None:
        lnx_grid = np.linspace(-8, 14, 300)
    x_grid = np.exp(lnx_grid)
    n_grid = n_mean * x_grid

    f_upper = level_fraction_vs_density(level_calc, beta, T, coll_partner_idx, n_grid, up_idx)
    dpdlnx = lognormal_dpdlnx(lnx_grid, sigma2)

    A_ul, g_u, g_l, lam = transition_info(level_calc, up_idx, low_idx)
    nu = C_CGS / lam
    beta_ul = beta[up_idx, low_idx]

    integrand_L = X_abund * beta_ul * A_ul * H_CGS * nu * f_upper * n_grid * dpdlnx
    integrand_M = MU_H2 * n_grid * dpdlnx

    L = np.trapz(integrand_L, lnx_grid)
    M = np.trapz(integrand_M, lnx_grid)
    
    return dict(L=L, M=M, lnx=lnx_grid, n=n_grid,
                dLdlnx=integrand_L, dMdlnx=integrand_M, f_upper=f_upper)
