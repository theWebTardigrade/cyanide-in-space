"""
config.py - Table 1 of Krumholz & Thompson (2007): fiducial model parameters
for "normal", "intermediate", and "starburst" galaxies, plus the CO/HCO+/HCN
abundances and 1-0 optical depths used for each case.
"""

GALAXY_CASES = {
    "normal":       dict(T=10, mach=30),
    "intermediate": dict(T=20, mach=50),
    "starburst":    dict(T=50, mach=80),
}

# X(species), tau_10 for each line, per galaxy case (Table 1)
LINE_PARAMS = {
    "CO": dict(
        X=dict(normal=2e-4, intermediate=4e-4, starburst=8e-4),
        tau10=dict(normal=10.0, intermediate=20.0, starburst=40.0),
    ),
    "HCO+": dict(
        X=dict(normal=2e-9, intermediate=4e-9, starburst=8e-9),
        tau10=dict(normal=0.5, intermediate=1.0, starburst=2.0),
    ),
    "HCN": dict(
        X=dict(normal=1e-8, intermediate=2e-8, starburst=4e-8),
        tau10=dict(normal=0.5, intermediate=1.0, starburst=2.0),
    ),
}

# LAMDA data file for each species, and which collision-partner table (by
# index into LevelCalculator.coll_tables) corresponds to H2. This is NOT
# always index 1: check with cloud_model.list_collision_partners() before
# trusting these for a new/updated data file.
#   CO:   [0] CO-pH2, [1] CO-oH2         -> use ortho-H2, index 1
#   HCO+: [0] pH2-HCO+, [1] oH2-HCO+     -> use ortho-H2, index 1
#   HCN:  [0] HCN-H2 (from He, scaled), [1] HCN-electrons -> use H2, index 0
MOLDATA_FILES = {"CO": "co.dat", "HCO+": "hco_.dat", "HCN": "hcn.dat"}
COLL_PARTNER_IDX = {"CO": 1, "HCO+": 1, "HCN": 0}

LINE_COLORS = {"CO": "#2166ac", "HCO+": "#238b45", "HCN": "#b2182b"}
LINE_LABELS = {"CO": "CO(1-0)", "HCO+": r"HCO$^+$(1-0)", "HCN": "HCN(1-0)"}


def get_cloud_kwargs(species, case):
    gp = GALAXY_CASES[case]
    lp = LINE_PARAMS[species]
    return dict(T=gp["T"], mach=gp["mach"], X_abund=lp["X"][case], tau_ref=lp["tau10"][case])
