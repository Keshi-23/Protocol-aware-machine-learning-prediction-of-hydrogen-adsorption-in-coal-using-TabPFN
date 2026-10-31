"""
H2 adsorption in coal — evaluation under three splitting protocols.

Replaces the single random 70:15:15 split with:

  RUN 1  three splitting protocols (random rows / held-out coal / held-out source)
  RUN 2  data-efficiency curve vs number of coal samples
  RUN 3  extrapolation in temperature and pressure
  RUN 4  physical consistency screening of predicted isotherms
  RUN 5  calibrated prediction intervals + P10/P50/P90 storage capacity vs depth

Models: TabPFN (defaults), XGBoost, Random Forest, MLP (all three re-tuned inside
each protocol with grouped inner folds), Langmuir (random-row protocol only).

Install:
    pip install tabpfn xgboost openpyxl scipy
    pip install "optuna==3.6.1"
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.optimize import curve_fit


# ═════════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═════════════════════════════════════════════════════════════════

FILE_PATH  = 'h2coal_dataset.xlsx'
SHEET_NAME = 'Sheet1'
OUT_DIR    = './results_protocols'

# Which runs to execute
RUN_1_PROTOCOLS   = False
RUN_2_EFFICIENCY  = False
RUN_3_EXTRAPOLATE = False
RUN_4_PHYSICS     = False
RUN_5_UNCERTAINTY = True
RUN_6_RELEVANCY   = False

# Column names. Set these explicitly if auto-detection picks the wrong ones.
COL_ID          = None            # no ID column in this file
DROP_COLS       = ['Link']        # non-numeric columns to exclude from features
COL_TARGET      = None            # None = last column
COL_TEMPERATURE = None            # None = auto-detect
COL_PRESSURE    = None            # None = auto-detect
COL_COMPOSITION = None            # None = auto-detect (moisture, ash, volatile, fixed carbon)
COL_SOURCE      = 'Author'        # source/reference column

# Rounding used when building sample_id from composition. Loosen if the source
# papers report composition to different precision and you get spurious splits.
COMPOSITION_DECIMALS = 2

# ── Isotherm monotonicity screening ──────────────────────────────
# Each isotherm (one coal composition at one temperature) is sorted by pressure
# and checked for uptake that decreases as pressure rises.
#
# MONO_MODE:
#   'off'           no screening
#   'flag'          report violations, keep all data (start here)
#   'drop_points'   remove only the measurements involved in a violation
#   'drop_isotherm' remove the whole isotherm
#
# Two tolerances must BOTH be exceeded before an isotherm is called a violation.
# This matters for your data: several sources report uptake to two decimal places,
# so a 0.01 mmol/g dip is reporting resolution rather than a bad measurement.
# Screening on the absolute drop alone would delete entire sources for rounding.
MONO_MODE     = 'off'
MONO_TOL_ABS  = 0.02    # mmol/g — a drop no larger than this is treated as noise
MONO_TOL_REL  = 0.10    # fraction of that isotherm's maximum uptake
MONO_MIN_PTS  = 3       # isotherms shorter than this cannot be screened meaningfully

# Excess sorption is physically allowed to peak and decline once the free-gas
# density rises enough. Set this to exempt the high-pressure tail from the check
# so that real turnover is not mistaken for bad data. None = check everywhere.
MONO_EXEMPT_ABOVE_MPA = None    # e.g. 12.0

N_FOLDS       = 5
SEED          = 42

# ── Models ───────────────────────────────────────────────────────
# TabPFN is used at its defaults; the other three are re-tuned inside every
# protocol with inner folds that respect the same grouping as the outer split.
# Tuning them on leaky inner folds would be the same mistake this paper is about.
MODELS        = ["TabPFN", "XGB", "RF", "MLP"]
TUNE_MODELS   = {"XGB": True, "RF": True, "MLP": True}
TUNE_TRIALS   = {"XGB": 50,   "RF": 30,   "MLP": 30}
TUNE_TABPFN   = False    # TabPFN needs no tuning, which is the point of the paper

MODEL_COLORS  = {"TabPFN": "#2E86AB", "XGB": "#E63946",
                 "RF": "#E9A03B", "MLP": "#6A4C93", "Langmuir": "#2A9D8F"}

# ── Monte Carlo settings (RUN 5c) ────────────────────────────────
MC_N            = 3000     # rows to generate
MC_DESIGN       = 'lhs'    # 'lhs' (Latin hypercube) or 'random'
MC_RANGE_EXPAND = 0.20     # widen each input range by +/- 20% of its span
MC_RENORMALISE  = True     # rescale M+A+V+FC to 100 wt% after sampling
MC_MODEL        = 'TabPFN' # model used for the Monte Carlo predictions
MC_PRESSURES    = [0.1, 1.0, 5.0, 10.0, 15.0, 18.0]   # pressures for the CDF panel

# TabPFN configuration (no tuning; these are the published defaults)
TABPFN_CONFIG = {
    "n_estimators"          : 12,
    "softmax_temperature"   : 0.9,
    "average_before_softmax": False,
    "output_type"           : "mean",
}

# XGB fallback if TUNE_XGB = False — these are the Table 2 values from the
# submitted manuscript. Note they were selected on a validation set that shared
# isotherms with the training set, so they should not be trusted under the
# grouped protocols.
XGB_DEFAULT = dict(
    n_estimators=4482, max_depth=46, learning_rate=0.1487,
    subsample=0.5378, colsample_bytree=0.5317, min_child_weight=10,
    gamma=1.38e-4, reg_alpha=7.15e-6, reg_lambda=0.7237,
    tree_method="hist", random_state=SEED, n_jobs=-1,
)

# ── Section 3.8 reservoir settings ────────────────────────────────
GEOTHERMAL_GRADIENT = 28.6      # K/km
PORE_PRESSURE_GRAD  = 8.5       # MPa/km
OVERBURDEN_GRAD     = 25.0      # MPa/km
SURFACE_TEMP        = 303.15    # K
SKELETON_DENSITY    = 1400.0    # kg/m3
POROSITY_0          = 0.05      # initial porosity, replace with your value
PORE_COMPRESSIBILITY= 0.02      # 1/MPa, replace with your value
DEPTH_MAX           = 2000.0    # m

M_H2      = 2.01588             # g/mol
R_S_H2    = 4124.2              # J/(kg K), specific gas constant
B_ABEL    = 7.691e-3            # m3/kg, Abel-Noble co-volume for H2

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════
#  TabPFN import and version handling
# ═════════════════════════════════════════════════════════════════

import tabpfn as _tabpfn_module
_tabpfn_version = getattr(_tabpfn_module, "__version__", "0")
_tabpfn_v2 = int(_tabpfn_version.split(".")[0]) >= 2
if not _tabpfn_v2:
    raise RuntimeError(
        "This script needs TabPFN v2 or later for the regression head and the "
        "quantile outputs used in RUN 5.  pip install -U tabpfn"
    )
from tabpfn import TabPFNRegressor

VALID_INF_CFG_FIELDS = set()
for _path in ("tabpfn.inference_config", "tabpfn.constants"):
    try:
        mod = __import__(_path, fromlist=["*"])
        cfg_cls = getattr(mod, "InferenceConfig", None) or getattr(mod, "ModelInterfaceConfig", None)
        if cfg_cls is not None:
            VALID_INF_CFG_FIELDS = set(cfg_cls.model_fields.keys())
            break
    except Exception:
        continue

print(f"TabPFN {_tabpfn_version} | {len(VALID_INF_CFG_FIELDS)} InferenceConfig fields")

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("pip install xgboost")


def build_tabpfn(config=None):
    cfg = dict(TABPFN_CONFIG if config is None else config)
    inf_cfg = {k: v for k, v in cfg.items()
               if k in VALID_INF_CFG_FIELDS and v is not None}
    return TabPFNRegressor(
        n_estimators              = cfg.get("n_estimators", 8),
        softmax_temperature       = cfg.get("softmax_temperature", 0.2),
        average_before_softmax    = cfg.get("average_before_softmax", True),
        ignore_pretraining_limits = True,
        inference_config          = inf_cfg if inf_cfg else None,
        random_state              = SEED,
    )


def tabpfn_predict(model, X, out_type="mean"):
    try:
        p = model.predict(X, output_type=out_type)
        if isinstance(p, dict):
            p = p.get(out_type, p.get("mean"))
        return np.asarray(p).ravel()
    except Exception:
        return np.asarray(model.predict(X)).ravel()


def tabpfn_quantiles(model, X, qs=(0.05, 0.10, 0.50, 0.90, 0.95)):
    """Returns dict {q: array}. Falls back to None if the version cannot do it."""
    try:
        out = model.predict(X, output_type="quantiles", quantiles=list(qs))
        return {q: np.asarray(out[i]).ravel() for i, q in enumerate(qs)}
    except Exception as e:
        print(f"  quantile prediction unavailable: {e}")
        return None


# ═════════════════════════════════════════════════════════════════
#  1.  Load data and build grouping keys
# ═════════════════════════════════════════════════════════════════

df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df_raw.columns = [str(c).strip() for c in df_raw.columns]   # headers have trailing spaces
df_raw = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
print(f"\nLoaded {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print(f"Columns: {list(df_raw.columns)}")


def find_col(candidates, columns):
    for c in columns:
        low = str(c).lower().replace("_", " ").replace("%", "")
        for cand in candidates:
            if cand in low:
                return c
    return None


cols = list(df_raw.columns)

col_target = COL_TARGET or [c for c in cols if c != COL_ID][-1]
col_temp   = COL_TEMPERATURE or find_col(["temp", "t (k", "t(k"], cols)
col_press  = COL_PRESSURE   or find_col(["press", "p (mpa", "p(mpa"], cols)

if COL_COMPOSITION is not None:
    comp_cols = list(COL_COMPOSITION)
else:
    comp_cols = []
    for keys in (["moist"], ["ash"], ["volatile", "vc"], ["fixed", "fc"]):
        c = find_col(keys, cols)
        if c is not None and c not in comp_cols:
            comp_cols.append(c)

col_source = COL_SOURCE or find_col(["source", "reference", "ref", "author", "study"], cols)

print(f"\nTarget       : {col_target}")
print(f"Temperature  : {col_temp}")
print(f"Pressure     : {col_press}")
print(f"Composition  : {comp_cols}")
print(f"Source       : {col_source if col_source else 'NOT FOUND'}")

if len(comp_cols) < 4:
    raise ValueError(
        f"Only found {len(comp_cols)} composition columns: {comp_cols}. "
        "Set COL_COMPOSITION explicitly at the top of the script — the whole "
        "point of RUN 1 depends on grouping correctly."
    )

# ── sample_id: one per unique coal x moisture state ──────────────
comp_rounded = df_raw[comp_cols].round(COMPOSITION_DECIMALS)
sample_key   = comp_rounded.astype(str).agg("|".join, axis=1)
sample_id    = pd.factorize(sample_key)[0]
df_raw["sample_id"] = sample_id

n_samples = len(np.unique(sample_id))
print(f"\n{'='*60}")
print(f"  {len(df_raw)} measurements come from {n_samples} distinct coal samples")
print(f"  Median measurements per sample: {np.median(np.bincount(sample_id)):.0f}")
print(f"{'='*60}")
print("\nReport the sample count in the paper. It is the number that describes")
print("dataset size honestly; 781 describes the number of pressure steps.")

# ── isotherm_id: one per (coal sample x temperature) ─────────────
if col_temp is not None:
    iso_key = sample_key + "|" + df_raw[col_temp].astype(str)
    df_raw["isotherm_id"] = pd.factorize(iso_key)[0]
    n_iso = df_raw["isotherm_id"].nunique()
    print(f"  {n_iso} distinct isotherms (sample x temperature), "
          f"median {np.median(np.bincount(df_raw['isotherm_id'])):.0f} pressure steps each")


# ═════════════════════════════════════════════════════════════════
#  Isotherm monotonicity screening
# ═════════════════════════════════════════════════════════════════

def screen_isotherms(data, mode=MONO_MODE):
    """Check each isotherm for uptake that falls as pressure rises.

    Returns (cleaned_dataframe, report_dataframe).  The report is written to
    disk whatever the mode, so the screening is auditable and can be described
    in the paper without re-running anything.
    """
    if mode == "off" or "isotherm_id" not in data.columns or col_press is None:
        return data, pd.DataFrame()

    rep, drop_rows, drop_isos = [], [], []

    for iso, g in data.groupby("isotherm_id"):
        g = g.sort_values(col_press)
        P = g[col_press].values.astype(float)
        Y = g[col_target].values.astype(float)
        ymax = max(float(np.nanmax(Y)), 1e-12)

        if len(g) < MONO_MIN_PTS:
            rep.append(dict(isotherm_id=iso, sample_id=g["sample_id"].iloc[0],
                            source=(g[col_source].iloc[0] if col_source else ""),
                            temperature=g[col_temp].iloc[0] if col_temp else np.nan,
                            n_points=len(g), n_drops=0, worst_abs=0.0, worst_rel=0.0,
                            verdict="too short to screen", action="kept"))
            continue

        d = np.diff(Y)
        step_P = P[1:]                       # pressure at the end of each step
        neg = d < 0
        if MONO_EXEMPT_ABOVE_MPA is not None:
            neg &= step_P <= MONO_EXEMPT_ABOVE_MPA

        worst_abs = float(-d[neg].min()) if neg.any() else 0.0
        worst_rel = worst_abs / ymax
        violates = (worst_abs > MONO_TOL_ABS) and (worst_rel > MONO_TOL_REL)

        action = "kept"
        if violates and mode == "drop_isotherm":
            drop_isos.append(iso); action = "isotherm removed"
        elif violates and mode == "drop_points":
            # remove the lower endpoint of every offending step
            bad_steps = np.where(neg & (-d > MONO_TOL_ABS) &
                                 (-d / ymax > MONO_TOL_REL))[0]
            idx = g.index.values[bad_steps + 1]
            drop_rows.extend(idx.tolist()); action = f"{len(idx)} point(s) removed"

        rep.append(dict(
            isotherm_id=iso, sample_id=g["sample_id"].iloc[0],
            source=(g[col_source].iloc[0] if col_source else ""),
            temperature=(g[col_temp].iloc[0] if col_temp else np.nan),
            n_points=len(g), max_uptake=ymax,
            n_drops=int(neg.sum()), worst_abs=worst_abs, worst_rel=worst_rel,
            worst_drop_at_MPa=(float(step_P[np.argmin(np.where(neg, d, 0))])
                               if neg.any() else np.nan),
            verdict=("violation" if violates
                     else ("within tolerance" if neg.any() else "monotonic")),
            action=action))

    report = pd.DataFrame(rep)

    if drop_isos:
        cleaned = data[~data["isotherm_id"].isin(drop_isos)].copy()
    elif drop_rows:
        cleaned = data.drop(index=drop_rows).copy()
    else:
        cleaned = data.copy()

    # ── console summary ──
    n_v = int((report.verdict == "violation").sum())
    n_t = int((report.verdict == "within tolerance").sum())
    print(f"\n── Isotherm monotonicity screening (mode = {mode}) ──")
    print(f"  {len(report)} isotherms checked")
    print(f"  {n_t} have small dips within tolerance "
          f"(<= {MONO_TOL_ABS} mmol/g or <= {MONO_TOL_REL:.0%} of peak) — kept")
    print(f"  {n_v} exceed both tolerances — {mode}")
    if n_v:
        v = report[report.verdict == "violation"]
        for _, r in v.iterrows():
            print(f"    isotherm {int(r.isotherm_id):3d} | sample {int(r.sample_id):2d} | "
                  f"{r.source} | T = {r.temperature} | "
                  f"drop {r.worst_abs:.3f} mmol/g ({r.worst_rel:.0%} of peak) "
                  f"at {r.worst_drop_at_MPa:.2f} MPa")
    print(f"  rows {len(data)} -> {len(cleaned)}   "
          f"coals {data['sample_id'].nunique()} -> {cleaned['sample_id'].nunique()}", end="")
    if col_source:
        print(f"   sources {data[col_source].nunique()} -> {cleaned[col_source].nunique()}")
    else:
        print()

    lost_sources = (set(data[col_source]) - set(cleaned[col_source])) if col_source else set()
    if lost_sources:
        print(f"  WARNING: screening removed every row from {sorted(lost_sources)}.")
        print("  That weakens the held-out-source protocol. Consider raising the")
        print("  tolerances or switching MONO_MODE to 'drop_points'.")

    return cleaned, report


df_raw, mono_report = screen_isotherms(df_raw)
if len(mono_report):
    mono_report.to_csv(os.path.join(OUT_DIR, "isotherm_screening_report.csv"), index=False)

    # diagnostic figure: every isotherm that failed, as measured
    failed = mono_report[mono_report.verdict == "violation"]["isotherm_id"].tolist()
    if failed:
        raw_again = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
        raw_again.columns = [str(c).strip() for c in raw_again.columns]
        k = raw_again[comp_cols].round(COMPOSITION_DECIMALS).astype(str).agg("|".join, axis=1)
        raw_again["isotherm_id"] = pd.factorize(k + "|" + raw_again[col_temp].astype(str))[0]
        n = len(failed)
        ncol = min(4, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for ax, iso in zip(axes.ravel(), failed):
            g = raw_again[raw_again.isotherm_id == iso].sort_values(col_press)
            ax.plot(g[col_press], g[col_target], "o-", color="#E53935")
            ax.set_title(f"isotherm {iso}", fontsize=9)
            ax.set_xlabel(col_press, fontsize=8); ax.set_ylabel(col_target, fontsize=8)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        plt.suptitle("Isotherms removed by monotonicity screening", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "isotherm_screening_removed.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # re-index groups after removal so counts and folds are clean
    df_raw = df_raw.reset_index(drop=True)
    df_raw["sample_id"] = pd.factorize(df_raw["sample_id"])[0]
    df_raw["isotherm_id"] = pd.factorize(df_raw["isotherm_id"])[0]
    n_samples = df_raw["sample_id"].nunique()
    print(f"\nAfter screening: {len(df_raw)} measurements, {n_samples} coal samples, "
          f"{df_raw['isotherm_id'].nunique()} isotherms")


# Cross-check against an existing ID column, if one looks like a curve index
if COL_ID is not None and COL_ID in df_raw.columns:
    sn_groups = pd.factorize(df_raw[COL_ID].astype(str))[0]
    n_sn = len(np.unique(sn_groups))
    print(f"\nExisting '{COL_ID}' column resolves to {n_sn} groups "
          f"(composition-based grouping gives {n_samples}).")
    if n_sn != n_samples:
        print("  These disagree. Check whether SN indexes isotherms, samples, or rows.")
        print("  If SN is already a per-sample ID, set it as the grouping variable below.")

# ── source_id ────────────────────────────────────────────────────
if col_source is not None:
    source_id = pd.factorize(df_raw[col_source].astype(str))[0]
    df_raw["source_id"] = source_id
    print(f"\n{len(np.unique(source_id))} distinct sources found.")
else:
    source_id = None
    template = (df_raw.groupby("sample_id")[comp_cols].first().reset_index())
    template["source"] = ""
    tpath = os.path.join(OUT_DIR, "source_labels_TEMPLATE.xlsx")
    template.to_excel(tpath, index=False)
    print(f"\nNo source column found. Wrote {tpath}")
    print("Fill in the 'source' column (one row per coal sample), add it back to")
    print("your data file, then re-run to enable the held-out-source protocol.")

# ── feature matrix ───────────────────────────────────────────────
drop_cols = [c for c in [COL_ID, col_source, "sample_id", "source_id", "isotherm_id"]
             if c is not None and c in df_raw.columns]
feature_names = [c for c in df_raw.columns if c not in drop_cols + [col_target]]

X = df_raw[feature_names].values.astype(np.float32)
y = df_raw[col_target].values.astype(np.float32)
groups_sample = df_raw["sample_id"].values

i_temp  = feature_names.index(col_temp)  if col_temp  in feature_names else None
i_press = feature_names.index(col_press) if col_press in feature_names else None

print(f"\nFeatures ({len(feature_names)}): {feature_names}")
if len(feature_names) != 6:
    print(f"  NOTE: manuscript text says six inputs; this file has {len(feature_names)}.")
    print("  Table 3 in the submitted version says nine. Reconcile before resubmission.")


# ═════════════════════════════════════════════════════════════════
#  2.  Shared helpers
# ═════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred):
    return {
        "R2"  : r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE" : float(mean_absolute_error(y_true, y_pred)),
    }


def fit_predict_tabpfn(X_tr, y_tr, X_te, config=None):
    sc = MinMaxScaler().fit(X_tr)
    m = build_tabpfn(config)
    m.fit(sc.transform(X_tr).astype(np.float32), y_tr)
    out_type = (config or TABPFN_CONFIG).get("output_type", "mean")
    return tabpfn_predict(m, sc.transform(X_te).astype(np.float32), out_type), m, sc


# ── generic four-model layer ─────────────────────────────────────
# Scaling is fitted on the training fold only, never on the full dataset.
SCALER_FOR = {"TabPFN": "minmax", "MLP": "standard", "XGB": None, "RF": None}

RF_DEFAULT  = dict(n_estimators=500, max_depth=None, min_samples_leaf=1,
                   random_state=SEED, n_jobs=-1)
MLP_DEFAULT = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                   learning_rate_init=1e-3, max_iter=3000, early_stopping=True,
                   validation_fraction=0.15, random_state=SEED)


def _make_scaler(name):
    kind = SCALER_FOR.get(name)
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    return None


def _make_estimator(name, params=None):
    if name == "XGB":
        return XGBRegressor(**dict(XGB_DEFAULT if params is None else params))
    if name == "RF":
        return RandomForestRegressor(**dict(RF_DEFAULT if params is None else params))
    if name == "MLP":
        return MLPRegressor(**dict(MLP_DEFAULT if params is None else params))
    raise ValueError(name)


def _suggest_params(name, trial):
    if name == "XGB":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 2000),
            max_depth        = trial.suggest_int("max_depth", 2, 10),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
            gamma            = trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            tree_method="hist", random_state=SEED, n_jobs=-1)
    if name == "RF":
        return dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 800),
            max_depth         = trial.suggest_int("max_depth", 2, 25),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features      = trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", 0.5, 1.0]),
            random_state=SEED, n_jobs=-1)
    if name == "MLP":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sizes = tuple(trial.suggest_categorical(f"h{i}", [16, 32, 64, 128])
                      for i in range(n_layers))
        return dict(
            hidden_layer_sizes = sizes,
            activation         = trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha              = trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            max_iter=3000, early_stopping=True, validation_fraction=0.15,
            random_state=SEED)
    raise ValueError(name)


def tune_model(name, X_tr, y_tr, groups_tr, grouped=True, n_trials=None):
    """Optuna search with inner folds that mirror the outer protocol's grouping."""
    if not TUNE_MODELS.get(name, False):
        return None
    if n_trials is None:
        n_trials = TUNE_TRIALS.get(name, 30)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    use_group = grouped and groups_tr is not None and len(np.unique(groups_tr)) >= 2
    n_inner = min(3, len(np.unique(groups_tr))) if use_group else 3
    splitter = (GroupKFold(n_splits=n_inner) if use_group
                else KFold(n_splits=3, shuffle=True, random_state=SEED))
    split_args = (X_tr, y_tr, groups_tr) if use_group else (X_tr,)

    def objective(trial):
        p = _suggest_params(name, trial)
        errs = []
        for tr, vl in splitter.split(*split_args):
            sc = _make_scaler(name)
            Xa, Xb = X_tr[tr], X_tr[vl]
            if sc is not None:
                Xa = sc.fit_transform(Xa); Xb = sc.transform(Xb)
            m = _make_estimator(name, p)
            m.fit(Xa, y_tr[tr])
            errs.append(mean_squared_error(y_tr[vl], m.predict(Xb)))
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if name == "MLP":
        bp = study.best_params
        sizes = tuple(bp[f"h{i}"] for i in range(bp["n_layers"]))
        return dict(hidden_layer_sizes=sizes, activation=bp["activation"],
                    alpha=bp["alpha"], learning_rate_init=bp["lr"],
                    max_iter=3000, early_stopping=True,
                    validation_fraction=0.15, random_state=SEED)
    best = dict(study.best_params)
    if name == "XGB":
        best.update(tree_method="hist", random_state=SEED, n_jobs=-1)
    else:
        best.update(random_state=SEED, n_jobs=-1)
    return best


def fit_model(name, X_tr, y_tr, groups_tr=None, grouped=True, params=None,
              tune=True, n_trials=None):
    """Fit any of the four models. Returns (fitted_model, scaler)."""
    sc = _make_scaler(name)
    Xa = X_tr if sc is None else sc.fit_transform(X_tr)
    if name == "TabPFN":
        m = build_tabpfn()
        m.fit(np.asarray(Xa, dtype=np.float32), y_tr)
        return m, sc
    if params is None and tune:
        params = tune_model(name, X_tr, y_tr, groups_tr, grouped, n_trials)
    m = _make_estimator(name, params)
    m.fit(Xa, y_tr)
    return m, sc


def predict_model(name, model, scaler, X_te):
    Xb = X_te if scaler is None else scaler.transform(X_te)
    if name == "TabPFN":
        return tabpfn_predict(model, np.asarray(Xb, dtype=np.float32))
    return np.asarray(model.predict(Xb)).ravel()


def fit_predict(name, X_tr, y_tr, X_te, groups_tr=None, grouped=True,
                params=None, tune=True, n_trials=None):
    m, sc = fit_model(name, X_tr, y_tr, groups_tr, grouped, params, tune, n_trials)
    return predict_model(name, m, sc, X_te), m, sc


# kept for RUN 5, which is TabPFN-specific
def fit_predict_xgb(X_tr, y_tr, X_te, params=None):
    m = XGBRegressor(**dict(XGB_DEFAULT if params is None else params))
    m.fit(X_tr, y_tr)
    return m.predict(X_te), m


# ═════════════════════════════════════════════════════════════════
#  RUN 1 — three splitting protocols
# ═════════════════════════════════════════════════════════════════

def langmuir(P, nL, PL):
    return nL * P / (P + PL)


def langmuir_random_split(train_idx, test_idx):
    """Fit Langmuir per isotherm using TRAINING rows only, predict held-out rows.

    This is the honest version of the benchmark. The submitted manuscript fits
    each isotherm on all of its points and then reports 'test' R2 on points that
    were part of the fit, which is fit quality, not prediction.
    """
    if i_press is None:
        return None
    preds = np.full(len(test_idx), np.nan)
    P_all = X[:, i_press]
    for s in np.unique(groups_sample):
        tr = train_idx[groups_sample[train_idx] == s]
        te_mask = groups_sample[test_idx] == s
        if te_mask.sum() == 0:
            continue
        if len(tr) < 3:
            continue                      # cannot fit two parameters
        try:
            popt, _ = curve_fit(langmuir, P_all[tr], y[tr],
                                p0=[max(y[tr].max(), 1e-3), 1.0],
                                bounds=([0, 1e-6], [np.inf, np.inf]),
                                maxfev=20000)
            preds[te_mask] = langmuir(P_all[test_idx[te_mask]], *popt)
        except Exception:
            continue
    return preds


def run_protocol(name, splitter, split_args, grouped, tune_groups=True):
    print(f"\n{'─'*60}\nProtocol: {name}\n{'─'*60}")
    rows = []
    fold_preds = []

    for fold, (tr, te) in enumerate(splitter.split(*split_args), 1):
        g_tr = groups_sample[tr] if tune_groups else None
        preds, line = {}, []

        for mname in MODELS:
            p, _, _ = fit_predict(mname, X[tr], y[tr], X[te],
                                  groups_tr=g_tr, grouped=grouped and tune_groups)
            preds[mname] = p
            mm = metrics(y[te], p)
            rows.append(dict(protocol=name, fold=fold, model=mname,
                             n_train=len(tr), n_test=len(te), **mm))
            line.append(f"{mname} R2={mm['R2']:.4f}")

        # Langmuir only makes sense when isotherms are shared across the split
        if not grouped:
            p_lan = langmuir_random_split(tr, te)
            if p_lan is not None:
                ok = ~np.isnan(p_lan)
                if ok.sum() > 5:
                    m_lan = metrics(y[te][ok], p_lan[ok])
                    m_lan["n_test"] = int(ok.sum())
                    rows.append(dict(protocol=name, fold=fold, model="Langmuir",
                                     n_train=len(tr), **m_lan))
                    line.append(f"Langmuir R2={m_lan['R2']:.4f}")

        fp = pd.DataFrame({"protocol": name, "fold": fold, "row_idx": te,
                           "sample_id": groups_sample[te], "actual": y[te]})
        for mname, p in preds.items():
            fp[mname] = p
        fold_preds.append(fp)

        print(f"  fold {fold}: " + "  ".join(line))

    return pd.DataFrame(rows), pd.concat(fold_preds, ignore_index=True)


protocol_results, protocol_preds = [], []

if RUN_1_PROTOCOLS:
    print("\n" + "="*60)
    print("RUN 1 — splitting protocols")
    print("="*60)

    # (a) the original single 70:15:15 random split, for continuity with the
    #     submitted manuscript
    idx = np.arange(len(X))
    tr_full, te_single = train_test_split(idx, test_size=0.15, random_state=SEED)
    tr_single, va_single = train_test_split(tr_full, test_size=0.1765, random_state=SEED)
    print("\nSingle random 70:15:15 split (reproduces the submitted numbers):")
    single_rows = []
    for mname in MODELS:
        p_s, _, _ = fit_predict(mname, X[tr_single], y[tr_single], X[te_single],
                                groups_tr=groups_sample[tr_single], grouped=False)
        mm = metrics(y[te_single], p_s)
        single_rows.append(dict(model=mname, **mm))
        print(f"  {mname:7s} test: R2={mm['R2']:.4f}  RMSE={mm['RMSE']:.4f}  MAE={mm['MAE']:.4f}")
    pd.DataFrame(single_rows).to_csv(
        os.path.join(OUT_DIR, "run1_single_random_split.csv"), index=False)

    # (b) random rows, 5-fold
    r, p = run_protocol("A_random_rows",
                        KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
                        (X,), grouped=False, tune_groups=False)
    protocol_results.append(r); protocol_preds.append(p)

    # (b2) leakage diagnostic — how much of the random test set is not really new
    if "isotherm_id" in df_raw.columns:
        iso = df_raw["isotherm_id"].values
        fr = []
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
            seen = set(iso[tr])
            fr.append(np.mean([i in seen for i in iso[te]]))
        print(f"\nLeakage diagnostic: {np.mean(fr)*100:.1f}% of test rows under random "
              f"splitting\nbelong to an isotherm that also appears in the training set.")

    # (b3) held-out isotherm (sample x temperature)
    if "isotherm_id" in df_raw.columns:
        r, p = run_protocol("B_holdout_isotherm",
                            GroupKFold(n_splits=N_FOLDS),
                            (X, y, df_raw["isotherm_id"].values), grouped=True)
        protocol_results.append(r); protocol_preds.append(p)

    # (c) held-out coal sample
    n_g = len(np.unique(groups_sample))
    r, p = run_protocol("C_holdout_sample",
                        GroupKFold(n_splits=min(N_FOLDS, n_g)),
                        (X, y, groups_sample), grouped=True)
    protocol_results.append(r); protocol_preds.append(p)

    # (d) held-out source
    if source_id is not None:
        n_s = len(np.unique(source_id))
        if n_s >= 2:
            r, p = run_protocol("D_holdout_source",
                                GroupKFold(n_splits=min(N_FOLDS, n_s)),
                                (X, y, source_id), grouped=True)
            protocol_results.append(r); protocol_preds.append(p)
    else:
        print("\nProtocol D skipped — no source column. See the template written above.")

    df_prot = pd.concat(protocol_results, ignore_index=True)
    summary = (df_prot.groupby(["protocol", "model"])[["R2", "RMSE", "MAE"]]
                      .agg(["mean", "std"]).round(4))
    print("\n" + "="*60)
    print("PROTOCOL SUMMARY  (this table goes in the paper)")
    print("="*60)
    print(summary)

    print("\nThe difference between protocol A and protocol B is the headline result.")
    print("Note that Langmuir has no entry under B or C: for a coal that was never")
    print("measured there is no nL or PL to use. State this in the paper — it is the")
    print("clearest argument for why a data-driven model is needed at all.")

    df_prot.to_csv(os.path.join(OUT_DIR, "run1_protocol_metrics.csv"), index=False)
    pd.concat(protocol_preds, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "run1_protocol_predictions.csv"), index=False)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        piv = df_prot.pivot_table(index="protocol", columns="model",
                                  values=metric, aggfunc="mean")
        err = df_prot.pivot_table(index="protocol", columns="model",
                                  values=metric, aggfunc="std")
        piv.plot(kind="bar", yerr=err, ax=ax, capsize=3, edgecolor="k", rot=20)
        ax.set_ylabel(metric); ax.set_title(metric); ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Model performance under three splitting protocols", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "run1_protocols.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════
#  RUN 2 — data-efficiency curve (samples, not rows)
# ═════════════════════════════════════════════════════════════════

if RUN_2_EFFICIENCY:
    print("\n" + "="*60)
    print("RUN 2 — data efficiency vs number of coal samples")
    print("="*60)

    uniq = np.unique(groups_sample)
    rng = np.random.default_rng(SEED)
    train_sizes = [s for s in [5, 10, 20, 40, 60] if s < len(uniq) - 3]
    train_sizes.append(len(uniq) - max(3, int(0.2 * len(uniq))))
    N_REPEATS = 5

    rows = []
    for rep in range(N_REPEATS):
        perm = rng.permutation(uniq)
        n_hold = max(3, int(0.2 * len(uniq)))
        hold, pool = perm[:n_hold], perm[n_hold:]
        te = np.isin(groups_sample, hold)

        for n in train_sizes:
            if n > len(pool):
                continue
            chosen = pool[:n]
            tr = np.isin(groups_sample, chosen)
            if tr.sum() < 10:
                continue

            line = []
            for mname in MODELS:
                # fewer trials here — this loop runs many times
                p, _, _ = fit_predict(mname, X[tr], y[tr], X[te],
                                      groups_tr=groups_sample[tr], grouped=True,
                                      n_trials=max(10, TUNE_TRIALS.get(mname, 30) // 3))
                mm = metrics(y[te], p)
                rows.append(dict(repeat=rep, n_samples=n, n_rows=int(tr.sum()),
                                 model=mname, **mm))
                line.append(f"{mname} R2={mm['R2']:.3f}")
            print(f"  rep {rep} n={n:3d} samples ({tr.sum():4d} rows): " + "  ".join(line))

    df_eff = pd.DataFrame(rows)
    df_eff.to_csv(os.path.join(OUT_DIR, "run2_data_efficiency.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        sub = df_eff[df_eff.model == model].groupby("n_samples")["R2"]
        if len(sub) == 0:
            continue
        ax.errorbar(sub.mean().index, sub.mean().values, yerr=sub.std().values,
                    marker="o", capsize=4, label=model,
                    color=MODEL_COLORS.get(model))
    ax.set_xlabel("Number of coal samples in training set")
    ax.set_ylabel("R² on held-out coal samples")
    ax.set_title("How many measured coals do you actually need?", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "run2_data_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\nIf TabPFN holds up where XGB collapses, that gap is a transferable")
    print("finding for subsurface ML, not just a model preference.")


# ═════════════════════════════════════════════════════════════════
#  RUN 3 — extrapolation in temperature and pressure
# ═════════════════════════════════════════════════════════════════

if RUN_3_EXTRAPOLATE:
    print("\n" + "="*60)
    print("RUN 3 — extrapolation")
    print("="*60)

    rows = []
    fitted = {}          # case label -> {model: (fitted_model, scaler)}

    def extrapolation_case(label, mask_train, mask_test, keep_models=False):
        if mask_train.sum() < 30 or mask_test.sum() < 10:
            print(f"  {label}: skipped (train {mask_train.sum()}, test {mask_test.sum()})")
            return
        line = []
        store = {}
        for mname in MODELS:
            m, sc = fit_model(mname, X[mask_train], y[mask_train],
                              groups_tr=groups_sample[mask_train], grouped=True)
            p = predict_model(mname, m, sc, X[mask_test])
            mm = metrics(y[mask_test], p)
            rows.append(dict(case=label, model=mname, n_train=int(mask_train.sum()),
                             n_test=int(mask_test.sum()), **mm))
            line.append(f"{mname} R2={mm['R2']:.3f}")
            store[mname] = (m, sc)
        if keep_models:
            fitted[label] = store
        print(f"  {label}:  " + "  ".join(line))

        # interpolation control — same test-set size, drawn at random
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(X))
        te_c = rng.choice(idx, size=int(mask_test.sum()), replace=False)
        tr_c = np.setdiff1d(idx, te_c)
        line = []
        for mname in MODELS:
            p_c, _, _ = fit_predict(mname, X[tr_c], y[tr_c], X[te_c],
                                    groups_tr=groups_sample[tr_c], grouped=False)
            mm = metrics(y[te_c], p_c)
            rows.append(dict(case=label + " [random control]", model=mname,
                             n_train=len(tr_c), n_test=len(te_c), **mm))
            line.append(f"{mname} R2={mm['R2']:.3f}")
        print(f"    random control:  " + "  ".join(line))

    # ── temperature extrapolation, both directions ───────────────
    T_LOW_CUT, T_HIGH_CUT = 333.2, 303.2
    if i_temp is not None:
        T = X[:, i_temp]
        # interpolate up: train on the cold end, predict the hottest isotherms
        extrapolation_case(f"T <= {T_LOW_CUT:.0f} K -> T > {T_LOW_CUT:.0f} K",
                           T <= T_LOW_CUT, T > T_LOW_CUT, keep_models=True)
        # interpolate down: train on the hot end, predict the coldest isotherms
        extrapolation_case(f"T >= {T_HIGH_CUT:.0f} K -> T < {T_HIGH_CUT:.0f} K",
                           T >= T_HIGH_CUT, T < T_HIGH_CUT)

    # ── pressure extrapolation, both directions ──────────────────
    P_HI_CUT, P_LO_CUT = 12.0, 5.0
    if i_press is not None:
        P = X[:, i_press]
        extrapolation_case(f"P <= {P_HI_CUT:.0f} MPa -> P > {P_HI_CUT:.0f} MPa",
                           P <= P_HI_CUT, P > P_HI_CUT, keep_models=True)
        extrapolation_case(f"P >= {P_LO_CUT:.0f} MPa -> P < {P_LO_CUT:.0f} MPa",
                           P >= P_LO_CUT, P < P_LO_CUT)

    if rows:
        df_ex = pd.DataFrame(rows)
        df_ex.to_csv(os.path.join(OUT_DIR, "run3_extrapolation.csv"), index=False)
        print("\n" + df_ex.pivot_table(index="case", columns="model",
                                       values="R2").round(3).to_string())
        print("\nThe gap between each case and its random control is the extrapolation")
        print("penalty. Reservoir screening always extrapolates, so this belongs in")
        print("the paper even if the numbers are unflattering.")

    # ═════════════════════════════════════════════════════════════
    #  RUN 3b — does each model reproduce the physics outside its
    #           training range?  Temperature and pressure sweeps.
    # ═════════════════════════════════════════════════════════════
    if i_temp is not None and i_press is not None and fitted:

        # representative coal: the sample with the most measurements
        sid_ref = int(pd.Series(groups_sample).value_counts().idxmax())
        base = df_raw.loc[df_raw.sample_id == sid_ref,
                          feature_names].median().values.astype(np.float32)
        T_obs = np.unique(X[:, i_temp])
        P_grid = np.linspace(0.0, 18.0, 80, dtype=np.float32)
        T_grid = np.linspace(float(T_obs.min()), float(T_obs.max()), 60, dtype=np.float32)

        def grid_at(T_vals, P_vals):
            n = max(np.size(T_vals), np.size(P_vals))
            g = np.tile(base, (n, 1))
            g[:, i_temp] = T_vals
            g[:, i_press] = P_vals
            return g

        # ── (i) isotherms predicted by a model trained only on T <= cut ──
        case_T = f"T <= {T_LOW_CUT:.0f} K -> T > {T_LOW_CUT:.0f} K"
        T_show = np.array([273.15, 303.15, 333.15, 363.15], dtype=np.float32)
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS), 4.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, mname in zip(axes, MODELS):
            m, sc = fitted[case_T][mname]
            for T_val in T_show:
                pred = predict_model(mname, m, sc,
                                     grid_at(np.full_like(P_grid, T_val), P_grid))
                held = T_val > T_LOW_CUT
                ax.plot(P_grid, pred, ls="--" if held else "-", lw=2.0,
                        label=f"{T_val:.0f} K" + (" (held out)" if held else ""))
            g_ref = df_raw[df_raw.sample_id == sid_ref]
            ax.scatter(g_ref[col_press], g_ref[col_target], s=16, c="k",
                       alpha=0.5, zorder=5, label="measured (this coal)")
            ax.set_xlabel("Pressure (MPa)")
            ax.set_title(mname, color=MODEL_COLORS.get(mname))
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axes[0].set_ylabel("Predicted H$_2$ adsorption (mmol/g)")
        plt.suptitle(f"Isotherms predicted after training only on T $\\leq$ "
                     f"{T_LOW_CUT:.0f} K — dashed curves are extrapolated",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "run3_temperature_extrapolation.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ── (ii) uptake vs temperature at fixed pressures ────────
        P_show = [2.0, 6.0, 12.0, 18.0]
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS), 4.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        mono_rows = []
        for ax, mname in zip(axes, MODELS):
            m, sc = fitted[case_T][mname]
            for P_val in P_show:
                pred = predict_model(mname, m, sc,
                                     grid_at(T_grid, np.full_like(T_grid, P_val)))
                ax.plot(T_grid, pred, lw=2.0, label=f"{P_val:.0f} MPa")
                d = np.diff(pred)
                mono_rows.append(dict(
                    model=mname, pressure_MPa=P_val,
                    increasing_steps=int((d > 1e-4).sum()), n_steps=len(d),
                    pct_increasing=100 * float((d > 1e-4).mean()),
                    worst_rise=float(max(d.max(), 0.0))))
            ax.axvspan(T_LOW_CUT, T_grid.max(), color="grey", alpha=0.15)
            ax.text(0.97, 0.95, "extrapolated", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="grey")
            ax.set_xlabel("Temperature (K)")
            ax.set_title(mname, color=MODEL_COLORS.get(mname))
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Predicted H$_2$ adsorption (mmol/g)")
        plt.suptitle("Temperature dependence at fixed pressure — physisorption "
                     "requires every curve to fall", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "run3_temperature_monotonicity.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        df_mono_T = pd.DataFrame(mono_rows)
        df_mono_T.to_csv(os.path.join(OUT_DIR, "run3_temperature_monotonicity.csv"),
                         index=False)
        print("\n  Temperature monotonicity of the T-extrapolating models "
              "(fraction of the sweep where uptake rises with T):")
        print(df_mono_T.groupby("model")["pct_increasing"].mean().round(1).to_string())

        # ── (iii) pressure sweeps from the P-extrapolating models ─
        case_P = f"P <= {P_HI_CUT:.0f} MPa -> P > {P_HI_CUT:.0f} MPa"
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS), 4.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, mname in zip(axes, MODELS):
            m, sc = fitted[case_P][mname]
            for T_val in T_show:
                pred = predict_model(mname, m, sc,
                                     grid_at(np.full_like(P_grid, T_val), P_grid))
                ax.plot(P_grid, pred, lw=2.0, label=f"{T_val:.0f} K")
            ax.axvspan(P_HI_CUT, 25.0, color="grey", alpha=0.15)
            ax.text(0.97, 0.05, "extrapolated", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="grey")
            ax.set_xlabel("Pressure (MPa)")
            ax.set_title(mname, color=MODEL_COLORS.get(mname))
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Predicted H$_2$ adsorption (mmol/g)")
        plt.suptitle(f"Isotherms predicted after training only on P $\\leq$ "
                     f"{P_HI_CUT:.0f} MPa", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "run3_pressure_extrapolation.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        print("\n  Tree ensembles cannot extrapolate: beyond the training range they")
        print("  return the last leaf value, so the isotherm goes flat. Watch for that")
        print("  in the shaded regions — it is a clearer argument than any R².")


# ═════════════════════════════════════════════════════════════════
#  RUN 4 — physical consistency screening
# ═════════════════════════════════════════════════════════════════

if RUN_4_PHYSICS:
    print("\n" + "="*60)
    print("RUN 4 — physical consistency of predicted isotherms")
    print("="*60)

    if i_press is None or i_temp is None:
        print("  Needs both a pressure and a temperature column. Skipped.")
    else:
        # 1e-4 mmol/g is below the reporting resolution of every source in the
        # dataset, so at that tolerance numerical wiggle counts as a violation
        # and almost every curve fails. Set it near the measurement resolution.
        TOL_MONO = 1e-3       # tolerance on a monotonicity violation, mmol/g
        TOL_ZERO = 5e-3       # tolerance on uptake at P -> 0, mmol/g

        # Build synthetic sweeps from real coal compositions
        comp_template = (df_raw.groupby("sample_id")[feature_names]
                               .median().reset_index(drop=True).values.astype(np.float32))
        P_grid = np.linspace(0.0, 18.0, 60, dtype=np.float32)
        T_grid = np.array([273.15, 303.15, 333.15, 363.15], dtype=np.float32)

        # train once on everything for the screening
        fitted_all = {}
        for mname in MODELS:
            fitted_all[mname] = fit_model(mname, X, y,
                                          groups_tr=groups_sample, grouped=True)

        def sweep(base_row, T_val, P_vals):
            grid = np.tile(base_row, (len(P_vals), 1))
            grid[:, i_temp]  = T_val
            grid[:, i_press] = P_vals
            return grid

        results = []
        for model_name in MODELS:
            mdl, msc = fitted_all[model_name]
            n_curves = 0
            v_p_mono = v_t_mono = v_zero = 0
            worst_dip = 0.0
            step_frac = []
            for base in comp_template:
                curves = {}
                for T_val in T_grid:
                    g = sweep(base, T_val, P_grid)
                    pred = predict_model(model_name, mdl, msc, g)
                    curves[float(T_val)] = pred
                    n_curves += 1

                    d = np.diff(pred)
                    step_frac.append(float((d < -TOL_MONO).mean()))
                    if np.any(d < -TOL_MONO):
                        v_p_mono += 1
                        worst_dip = max(worst_dip, float(-d.min()))
                    if abs(pred[0]) > TOL_ZERO:
                        v_zero += 1

                # temperature monotonicity at each pressure
                stacked = np.vstack([curves[float(t)] for t in T_grid])
                dT = np.diff(stacked, axis=0)
                if np.any(dT > TOL_MONO):
                    v_t_mono += 1

            results.append(dict(
                model=model_name,
                curves=n_curves,
                pressure_monotonicity_violations=v_p_mono,
                pct_pressure=100 * v_p_mono / max(n_curves, 1),
                zero_pressure_violations=v_zero,
                pct_zero=100 * v_zero / max(n_curves, 1),
                temperature_monotonicity_violations=v_t_mono,
                pct_temperature=100 * v_t_mono / max(len(comp_template), 1),
                worst_pressure_dip_mmol_g=worst_dip,
                pct_steps_decreasing=100 * float(np.mean(step_frac)) if step_frac else 0.0,
            ))
            print(f"  {model_name}: dP violations {v_p_mono}/{n_curves} curves "
                  f"({100*v_p_mono/max(n_curves,1):.1f}%; "
                  f"{100*np.mean(step_frac):.1f}% of steps), "
                  f"P->0 violations {v_zero}/{n_curves}, "
                  f"dT violations {v_t_mono}/{len(comp_template)}")

        df_phys = pd.DataFrame(results)
        df_phys.to_csv(os.path.join(OUT_DIR, "run4_physics_violations.csv"), index=False)

        # illustrative figure on one composition
        base = comp_template[0]
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS), 4.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, mname in zip(axes, MODELS):
            mdl, msc = fitted_all[mname]
            for T_val in T_grid:
                pred = predict_model(mname, mdl, msc, sweep(base, T_val, P_grid))
                ax.plot(P_grid, pred, label=f"T = {T_val:.0f} K")
            ax.set_xlabel("Pressure (MPa)")
            ax.set_title(mname, color=MODEL_COLORS.get(mname))
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Predicted H$_2$ adsorption (mmol/g)")
        plt.suptitle("Predicted isotherm shape — physical plausibility check",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "run4_isotherm_shape.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        print("\nTree ensembles usually produce step-like isotherms and a non-zero")
        print("intercept at P = 0. If the violation table shows that, it is a")
        print("selection criterion beyond R² and it ties the paper to the physics.")

        # ═════════════════════════════════════════════════════════
        #  RUN 4b — van 't Hoff / Clausius-Clapeyron check
        # ═════════════════════════════════════════════════════════
        # At fixed surface coverage n*, the Clausius-Clapeyron relation gives
        #
        #     (d ln P / d (1/T))_n*  =  ΔH_ads / R
        #
        # so a model that has learned the physics should produce straight lines
        # on ln P against 1/T, with a negative slope (exothermic physisorption).
        # Hu et al. [17] measured ΔH between −10.49 and −7.20 kJ/mol for coals of
        # varying rank, which is the number to compare against.
        print("\n" + "="*60)
        print("RUN 4b — van 't Hoff analysis (isosteric enthalpy of adsorption)")
        print("="*60)

        R_GAS   = 8.314                       # J mol-1 K-1
        T_VH    = np.array([273.15, 293.15, 313.15, 333.15, 353.15, 363.15],
                           dtype=np.float32)
        P_FINE  = np.linspace(0.05, 18.0, 600, dtype=np.float32)
        COVERAGES = [0.05, 0.10, 0.15, 0.20, 0.30]   # mmol/g
        HU_RANGE  = (-10.49, -7.20)                  # kJ/mol, Hu et al. [17]

        def pressure_at_coverage(mname, mdl, msc, base_row, T_val, n_star):
            """Invert the predicted isotherm to find P giving uptake n_star."""
            g = sweep(base_row, T_val, P_FINE)
            pred = predict_model(mname, mdl, msc, g)
            pred = np.maximum.accumulate(pred)     # enforce monotone for inversion
            if n_star < pred.min() or n_star > pred.max():
                return np.nan
            return float(np.interp(n_star, pred, P_FINE))

        vh_rows = []
        for mname in MODELS:
            mdl, msc = fitted_all[mname]
            for s_idx, base_row in enumerate(comp_template):
                for n_star in COVERAGES:
                    P_star = np.array([pressure_at_coverage(mname, mdl, msc,
                                                            base_row, T_val, n_star)
                                       for T_val in T_VH])
                    ok = np.isfinite(P_star) & (P_star > 0)
                    if ok.sum() < 4:
                        continue
                    x = 1.0 / T_VH[ok].astype(float)
                    z = np.log(P_star[ok])
                    slope, intercept = np.polyfit(x, z, 1)
                    z_hat = slope * x + intercept
                    ss_res = float(np.sum((z - z_hat) ** 2))
                    ss_tot = float(np.sum((z - z.mean()) ** 2))
                    lin_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
                    dH_kJ = R_GAS * slope / 1000.0        # kJ/mol; negative = exothermic
                    vh_rows.append(dict(
                        model=mname, sample_index=s_idx, coverage_mmol_g=n_star,
                        n_points=int(ok.sum()), slope=float(slope),
                        linearity_R2=float(lin_r2), dH_kJ_per_mol=float(dH_kJ),
                        exothermic=bool(dH_kJ < 0),
                        within_Hu_range=bool(HU_RANGE[0] <= dH_kJ <= HU_RANGE[1])))

        if vh_rows:
            df_vh = pd.DataFrame(vh_rows)
            df_vh.to_csv(os.path.join(OUT_DIR, "run4b_vant_hoff.csv"), index=False)

            summ = df_vh.groupby("model").agg(
                fits=("dH_kJ_per_mol", "size"),
                dH_mean=("dH_kJ_per_mol", "mean"),
                dH_sd=("dH_kJ_per_mol", "std"),
                median_linearity_R2=("linearity_R2", "median"),
                pct_exothermic=("exothermic", lambda v: 100 * v.mean()),
                pct_within_Hu=("within_Hu_range", lambda v: 100 * v.mean()),
            ).round(3)
            summ.to_csv(os.path.join(OUT_DIR, "run4b_vant_hoff_summary.csv"))
            print(summ.to_string())
            print(f"\n  Measured reference (Hu et al. [17]): "
                  f"{HU_RANGE[0]} to {HU_RANGE[1]} kJ/mol")
            print("  A model that has learned the physics gives straight lines")
            print("  (high linearity R²), a negative ΔH, and a magnitude in the")
            print("  physisorption range. Getting the sign right but the magnitude")
            print("  wrong is still worth reporting — it is a sharper discriminator")
            print("  between models than test R² is.")

            # ── van 't Hoff figure, one panel per model ──
            base_ref = comp_template[0]
            fig, axes = plt.subplots(1, len(MODELS),
                                     figsize=(4.4 * len(MODELS), 4.4), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, mname in zip(axes, MODELS):
                mdl, msc = fitted_all[mname]
                labels = []
                for n_star in COVERAGES:
                    P_star = np.array([pressure_at_coverage(mname, mdl, msc,
                                                            base_ref, T_val, n_star)
                                       for T_val in T_VH])
                    ok = np.isfinite(P_star) & (P_star > 0)
                    if ok.sum() < 4:
                        continue
                    x = 1000.0 / T_VH[ok].astype(float)      # 1000/T for readability
                    z = np.log(P_star[ok])
                    ax.plot(x, z, "o", ms=5)
                    sl, ic = np.polyfit(x, z, 1)
                    xf = np.linspace(x.min(), x.max(), 20)
                    ax.plot(xf, sl * xf + ic, "-", lw=1.4)
                    dH = R_GAS * (sl * 1000.0) / 1000.0
                    labels.append(f"n = {n_star:.2f}: ΔH = {dH:+.1f} kJ/mol")
                ax.set_xlabel("1000 / T  (K$^{-1}$)")
                ax.set_title(mname, color=MODEL_COLORS.get(mname))
                ax.grid(alpha=0.3)
                if labels:
                    ax.text(0.03, 0.03, "\n".join(labels), transform=ax.transAxes,
                            fontsize=7, va="bottom",
                            bbox=dict(fc="white", ec="grey", alpha=0.8))
            axes[0].set_ylabel("ln P  (P in MPa)")
            plt.suptitle("van 't Hoff plots at fixed coverage — slope gives the "
                         "isosteric enthalpy of adsorption", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "run4b_vant_hoff.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

            # ── enthalpy comparison bar chart against the measured range ──
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            means = df_vh.groupby("model")["dH_kJ_per_mol"].mean()
            sds   = df_vh.groupby("model")["dH_kJ_per_mol"].std()
            order = [m for m in MODELS if m in means.index]
            ax.bar(order, [means[m] for m in order],
                   yerr=[sds[m] for m in order], capsize=4, edgecolor="k",
                   color=[MODEL_COLORS.get(m) for m in order])
            ax.axhspan(HU_RANGE[0], HU_RANGE[1], color="grey", alpha=0.25,
                       label="measured range, Hu et al. [17]")
            ax.axhline(0, color="k", lw=1)
            ax.set_ylabel("Isosteric ΔH$_{ads}$ (kJ/mol)")
            ax.set_title("Enthalpy of adsorption implied by each model",
                         fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "run4b_enthalpy_comparison.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

    # ═════════════════════════════════════════════════════════════════
    # RUN 5 — random-split TabPFN intervals and depth-dependent inventory
    #
    # Requires existing:
    # X, y, feature_names, i_temp, i_press,
    # build_tabpfn, tabpfn_quantiles,
    # OUT_DIR, RUN_5_UNCERTAINTY,
    # DEPTH_MAX, SURFACE_TEMP, GEOTHERMAL_GRADIENT,
    # PORE_PRESSURE_GRAD, OVERBURDEN_GRAD,
    # POROSITY_0, PORE_COMPRESSIBILITY, SKELETON_DENSITY,
    # R_S_H2, B_ABEL, M_H2.
    # ═════════════════════════════════════════════════════════════════

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        mean_absolute_error,
    )

    RUN5_RANDOM_STATE = 42
    RUN5_TEST_SIZE = 0.20

    # To reproduce an EXISTING random-split result exactly, assign its
    # saved positional train/test indices here. Otherwise, the code
    # creates a reproducible 80:20 random split.
    RUN5_TRAIN_INDICES = None
    RUN5_TEST_INDICES = None

    COAL_PROPERTIES = {
        "Bi-1": {
            "FC": 57.77,
            "VM": 19.06,
            "A": 22.54,
            "M": 0.64,
        },
        "Bi-4": {
            "FC": 71.90,
            "VM": 13.02,
            "A": 14.46,
            "M": 0.62,
        },
    }

    PROXIMATE_COLUMNS = {
        "FC": "Fixed carbon (wt%)",
        "VM": "Volatile Content (wt%)",
        "A": "Ash Content (wt%)",
        "M": "Moisture Content (wt%)",
    }


    def h2_density(P_MPa, T_K):
        """Hydrogen density in kg/m³ using the specified Abel–Noble EOS."""
        pressure_pa = np.asarray(P_MPa, dtype=float) * 1e6
        temperature_k = np.asarray(T_K, dtype=float)

        if np.any(pressure_pa < 0):
            raise ValueError("Pressure must be nonnegative.")
        if np.any(temperature_k <= 0):
            raise ValueError("Temperature must be positive.")

        return pressure_pa / (
                R_S_H2 * temperature_k + B_ABEL * pressure_pa
        )


    def free_gas_capacity(P_MPa, T_K, porosity):
        """
        Bulk-gas pore-volume contribution, mmol H2/g coal.

        SKELETON_DENSITY: kg/m³
        M_H2: g/mol, approximately 2.01588
        """
        phi = np.asarray(porosity, dtype=float)

        if np.any((phi < 0) | (phi >= 1)):
            raise ValueError("Porosity must satisfy 0 <= porosity < 1.")
        if SKELETON_DENSITY <= 0:
            raise ValueError("SKELETON_DENSITY must be positive.")
        if not 2.0 < M_H2 < 2.1:
            raise ValueError("Use M_H2 in g/mol, e.g. 2.01588.")

        rho = h2_density(P_MPa, T_K)
        pore_volume_per_kg = phi / (
                (1.0 - phi) * SKELETON_DENSITY
        )
        mass_ratio = rho * pore_volume_per_kg

        return mass_ratio / M_H2 * 1000.0


    def checked_quantiles(model, inputs, probabilities):
        """Return finite, ordered quantiles; stop if unavailable."""
        result = tabpfn_quantiles(
            model,
            inputs,
            qs=probabilities,
        )

        if result is None:
            raise RuntimeError(
                "TabPFN quantile prediction failed. "
                "Check the installed regressor and quantile helper."
            )

        quantiles = {
            probability: np.asarray(
                result[probability], dtype=float
            ).reshape(-1)
            for probability in probabilities
        }

        matrix = np.column_stack([
            quantiles[probability]
            for probability in probabilities
        ])

        if matrix.shape[0] != len(inputs):
            raise ValueError("Quantile output length does not match inputs.")
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite quantile predictions detected.")
        if np.any(np.diff(matrix, axis=1) < -1e-7):
            raise ValueError("Crossing quantiles detected.")

        return quantiles


    if RUN_5_UNCERTAINTY:
        os.makedirs(OUT_DIR, exist_ok=True)

        print("\n" + "=" * 70)
        print("RUN 5 — random-split TabPFN intervals and Bi-1/Bi-4 inventory")
        print("=" * 70)

        X5 = np.asarray(X, dtype=float)
        y5 = np.asarray(y, dtype=float).reshape(-1)
        feature_names5 = list(feature_names)

        if X5.ndim != 2 or len(X5) != len(y5):
            raise ValueError("X and y must contain aligned observations.")
        if X5.shape[1] != len(feature_names5):
            raise ValueError("feature_names must match the columns of X.")
        if not np.isfinite(X5).all() or not np.isfinite(y5).all():
            raise ValueError("X and y must contain finite values.")
        if i_temp is None or i_press is None:
            raise ValueError("Temperature and pressure indices are required.")

        feature_indices = {}

        for component, column in PROXIMATE_COLUMNS.items():
            if column not in feature_names5:
                raise KeyError(f"Required feature missing: {column}")
            feature_indices[component] = feature_names5.index(column)

        # Every depth-grid feature must be explicitly specified.
        specified_indices = {
            int(i_temp),
            int(i_press),
            *feature_indices.values(),
        }

        if (
                len(specified_indices) != 6
                or specified_indices != set(range(X5.shape[1]))
        ):
            raise ValueError(
                "This block expects exactly six distinct features: "
                "temperature, pressure, moisture, ash, volatile matter "
                "and fixed carbon."
            )

        # ── 5a. Single random train/test split ────────────────────────
        all_indices = np.arange(len(y5))

        if RUN5_TRAIN_INDICES is None and RUN5_TEST_INDICES is None:
            tr, te = train_test_split(
                all_indices,
                test_size=RUN5_TEST_SIZE,
                random_state=RUN5_RANDOM_STATE,
                shuffle=True,
            )
        elif (
                RUN5_TRAIN_INDICES is not None
                and RUN5_TEST_INDICES is not None
        ):
            tr = np.asarray(RUN5_TRAIN_INDICES, dtype=int).reshape(-1)
            te = np.asarray(RUN5_TEST_INDICES, dtype=int).reshape(-1)
        else:
            raise ValueError(
                "Provide both saved train/test index arrays, or neither."
            )

        if len(tr) < 2 or len(te) < 2:
            raise ValueError("Training and test subsets require at least two rows.")

        combined_indices = np.concatenate([tr, te])

        if (
                len(combined_indices) != len(y5)
                or not np.array_equal(
            np.sort(combined_indices), all_indices
        )
        ):
            raise ValueError(
                "Train/test indices must partition all rows exactly once."
            )

        # Scaling is fitted on training inputs only.
        sc_random = MinMaxScaler().fit(X5[tr])

        X_train_scaled = sc_random.transform(X5[tr]).astype(np.float32)
        X_test_scaled = sc_random.transform(X5[te]).astype(np.float32)

        tab_random = build_tabpfn()
        tab_random.fit(X_train_scaled, y5[tr])

        print(
            f"  Random split: {len(tr)} training rows, "
            f"{len(te)} test rows"
        )

        # Export split identities for exact reproduction.
        split_labels = np.full(len(y5), "train", dtype=object)
        split_labels[te] = "test"

        df_split = pd.DataFrame({
            "row_idx": all_indices,
            "subset": split_labels,
        })

        # Mean point predictions are separate from median quantiles.
        prediction_frames = []
        metric_rows = []

        for subset, indices, scaled_inputs in [
            ("train", tr, X_train_scaled),
            ("test", te, X_test_scaled),
        ]:
            predicted_mean = np.asarray(
                tab_random.predict(
                    scaled_inputs,
                    output_type="mean",
                ),
                dtype=float,
            ).reshape(-1)

            if (
                    len(predicted_mean) != len(indices)
                    or not np.isfinite(predicted_mean).all()
            ):
                raise ValueError("Invalid mean predictions.")

            prediction_frames.append(pd.DataFrame({
                "protocol": "A_random_rows",
                "subset": subset,
                "row_idx": indices,
                "measured": y5[indices],
                "TabPFN_mean": predicted_mean,
            }))

            metric_rows.append({
                "protocol": "A_random_rows",
                "subset": subset,
                "n": len(indices),
                "R2": r2_score(y5[indices], predicted_mean),
                "RMSE": np.sqrt(
                    mean_squared_error(y5[indices], predicted_mean)
                ),
                "MAE": mean_absolute_error(
                    y5[indices], predicted_mean
                ),
            })

        df_predictions = pd.concat(
            prediction_frames, ignore_index=True
        )
        df_metrics = pd.DataFrame(metric_rows)

        q = checked_quantiles(
            tab_random,
            X_test_scaled,
            probabilities=(0.05, 0.10, 0.50, 0.90, 0.95),
        )

        covered90 = (
                (y5[te] >= q[0.05])
                & (y5[te] <= q[0.95])
        )
        covered80 = (
                (y5[te] >= q[0.10])
                & (y5[te] <= q[0.90])
        )

        coverage90 = float(covered90.mean())
        coverage80 = float(covered80.mean())
        width90 = float(np.mean(q[0.95] - q[0.05]))

        df_cov = pd.DataFrame([{
            "protocol": "A_random_rows",
            "n_train": len(tr),
            "n_test": len(te),
            "empirical_90_coverage": coverage90,
            "empirical_80_coverage": coverage80,
            "mean_90_width": width90,
            "calibration_adjustment_applied": False,
        }])

        dfq = pd.DataFrame({
            "protocol": "A_random_rows",
            "row_idx": te,
            "actual": y5[te],
            "q05": q[0.05],
            "q10": q[0.10],
            "q50": q[0.50],
            "q90": q[0.90],
            "q95": q[0.95],
        })

        print(f"  Test 90% interval coverage: {coverage90:.1%}")
        print(f"  Test 80% interval coverage: {coverage80:.1%}")
        print(f"  Mean 90% interval width: {width90:.5f} mmol/g")

        # ── Test prediction-interval plot ────────────────────────────
        ordered = dfq.sort_values("actual").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 4.5))

        ax.fill_between(
            ordered.index,
            ordered["q05"].to_numpy(),
            ordered["q95"].to_numpy(),
            alpha=0.25,
            color="#2E86AB",
            label="Nominal 90% prediction interval",
        )
        ax.plot(
            ordered.index,
            ordered["q50"],
            color="#2E86AB",
            lw=1.2,
            label="Median prediction",
        )
        ax.scatter(
            ordered.index,
            ordered["actual"],
            s=12,
            color="black",
            zorder=5,
            label="Measured",
        )

        ax.set_xlabel("Test observation, sorted by measured adsorption")
        ax.set_ylabel(r"H$_2$ adsorption (mmol/g)")
        ax.set_title(
            "TabPFN: 80:20 random-split test predictions\n"
            f"90% interval empirical coverage = {coverage90:.1%}"
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()

        for extension in ("png", "pdf"):
            fig.savefig(
                os.path.join(
                    OUT_DIR,
                    f"run5_random_intervals.{extension}",
                ),
                dpi=600,
                bbox_inches="tight",
            )
        plt.close(fig)

        # ── 5b. Depth-dependent Bi-1 and Bi-4 inventory ────────────────
        # IMPORTANT: reuse tab_random and sc_random.
        # There is no full-dataset refit.

        print("\n  Depth-dependent inventory: Bi-1 and Bi-4")

        depths = np.linspace(0.0, DEPTH_MAX, 101)

        T_d = (
                SURFACE_TEMP
                + GEOTHERMAL_GRADIENT * depths / 1000.0
        )
        P_d = PORE_PRESSURE_GRAD * depths / 1000.0
        S_d = OVERBURDEN_GRAD * depths / 1000.0

        sigma_eff = S_d - P_d
        porosity_d = POROSITY_0 * np.exp(
            -PORE_COMPRESSIBILITY * sigma_eff
        )

        gf = free_gas_capacity(P_d, T_d, porosity_d)

        training_min = X5[tr].min(axis=0)
        training_max = X5[tr].max(axis=0)

        depth_rows = []
        profile_audit = []

        for coal_name, properties in COAL_PROPERTIES.items():
            grid = np.zeros(
                (len(depths), X5.shape[1]),
                dtype=float,
            )

            grid[:, i_temp] = T_d
            grid[:, i_press] = P_d

            for component, column_index in feature_indices.items():
                grid[:, column_index] = properties[component]

            # Identify matching compositions for overlap reporting only.
            # These matches do not change the training/test split.
            composition_match = np.ones(len(y5), dtype=bool)

            for component, column_index in feature_indices.items():
                composition_match &= np.isclose(
                    X5[:, column_index],
                    properties[component],
                    atol=0.03,
                    rtol=0.0,
                )

            outside_training_range = (
                    (grid < training_min - 1e-8)
                    | (grid > training_max + 1e-8)
            ).any(axis=1)

            X_grid_scaled = sc_random.transform(grid).astype(np.float32)

            depth_q = checked_quantiles(
                tab_random,
                X_grid_scaled,
                probabilities=(0.10, 0.50, 0.90),
            )

            gex_p10 = depth_q[0.10]
            gex_p50 = depth_q[0.50]
            gex_p90 = depth_q[0.90]

            depth_rows.append(pd.DataFrame({
                "coal_name": coal_name,
                "protocol": "A_random_rows",
                "depth_m": depths,
                "T_K": T_d,
                "P_MPa": P_d,
                "effective_stress_MPa": sigma_eff,
                "porosity": porosity_d,
                "H2_density_kg_m3": h2_density(P_d, T_d),
                "FC_wt_percent": properties["FC"],
                "VM_wt_percent": properties["VM"],
                "Ash_wt_percent": properties["A"],
                "Moisture_wt_percent": properties["M"],
                "outside_training_feature_range": outside_training_range,
                "Gex_low": gex_p10,
                "Gex_mid": gex_p50,
                "Gex_high": gex_p90,
                "Gfree": gf,
                "UHS_low": gex_p10 + gf,
                "UHS_mid": gex_p50 + gf,
                "UHS_high": gex_p90 + gf,
            }))

            profile_audit.append({
                "coal_name": coal_name,
                "proximate_total_wt_percent": sum(properties.values()),
                "matching_composition_train_rows":
                    int(composition_match[tr].sum()),
                "matching_composition_test_rows":
                    int(composition_match[te].sum()),
                "depth_points_outside_training_feature_range":
                    int(outside_training_range.sum()),
            })

        df_depth = pd.concat(depth_rows, ignore_index=True)
        df_audit = pd.DataFrame(profile_audit)

        # ── Save numerical outputs ───────────────────────────────────
        output_tables = {
            "run5_random_split_indices": df_split,
            "run5_random_train_test_predictions": df_predictions,
            "run5_random_train_test_metrics": df_metrics,
            "run5_random_interval_coverage": df_cov,
            "run5_random_quantile_predictions": dfq,
            "run5_random_Bi1_Bi4_depth_capacity": df_depth,
            "run5_random_Bi1_Bi4_profile_audit": df_audit,
        }

        for filename, table in output_tables.items():
            table.to_csv(
                os.path.join(OUT_DIR, f"{filename}.csv"),
                index=False,
            )

        # Excel workbook for convenient plotting.
        with pd.ExcelWriter(
                os.path.join(OUT_DIR, "run5_random_Bi1_Bi4_results.xlsx"),
                engine="openpyxl",
        ) as writer:
            df_depth.to_excel(
                writer, sheet_name="Depth_inventory", index=False
            )
            df_predictions.to_excel(
                writer, sheet_name="Train_test_predictions", index=False
            )
            df_metrics.to_excel(
                writer, sheet_name="Performance", index=False
            )
            dfq.to_excel(
                writer, sheet_name="Test_quantiles", index=False
            )
            df_cov.to_excel(
                writer, sheet_name="Interval_coverage", index=False
            )
            df_audit.to_excel(
                writer, sheet_name="Profile_audit", index=False
            )
            df_split.to_excel(
                writer, sheet_name="Split_indices", index=False
            )

        # ── Depth plot: 1 row × 2 columns ─────────────────────────────
        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 7),
            sharey=True,
        )

        for ax, coal_name in zip(axes, COAL_PROPERTIES):
            d = (
                df_depth[df_depth["coal_name"] == coal_name]
                .sort_values("depth_m")
            )
            z = d["depth_m"].to_numpy()

            ax.fill_betweenx(
                z,
                d["Gex_low"].to_numpy(),
                d["Gex_high"].to_numpy(),
                alpha=0.25,
                color="green",
                label=r"$G_{\mathrm{ex}}$ Q0.10–Q0.90",
            )
            ax.plot(
                d["Gex_mid"], z,
                color="green", lw=1.8,
                label=r"$G_{\mathrm{ex}}$ median",
            )
            ax.plot(
                d["Gfree"], z,
                color="tab:blue", lw=1.8,
                label="Bulk-gas pore-volume term",
            )
            ax.fill_betweenx(
                z,
                d["UHS_low"].to_numpy(),
                d["UHS_high"].to_numpy(),
                alpha=0.20,
                color="red",
                label="Total Q0.10–Q0.90",
            )
            ax.plot(
                d["UHS_mid"], z,
                color="red", lw=1.8,
                label="Total median",
            )

            ax.set_xlabel(r"H$_2$ inventory (mmol/g coal)")
            ax.set_title(coal_name, fontweight="bold")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")

        # Set the shared depth-axis direction once.
        axes[0].set_ylim(DEPTH_MAX, 0)
        axes[0].set_ylabel("Depth (m)")

        fig.suptitle(
            "Depth-dependent inventory: random-split TabPFN model",
            fontweight="bold",
        )
        fig.tight_layout()

        for extension in ("png", "pdf"):
            fig.savefig(
                os.path.join(
                    OUT_DIR,
                    f"run5_random_Bi1_Bi4_depth_capacity.{extension}",
                ),
                dpi=600,
                bbox_inches="tight",
            )
        plt.close(fig)

        # ── Endpoint summary ─────────────────────────────────────────
        endpoints = (
            df_depth.sort_values("depth_m")
            .groupby("coal_name", sort=False)
            .tail(1)
        )

        for _, row in endpoints.iterrows():
            ratio = (
                row["Gex_mid"] / row["Gfree"]
                if row["Gfree"] > 0 else np.nan
            )

            print(
                f"\n  {row['coal_name']} at {row['depth_m']:.0f} m:"
                f"\n    Total median = {row['UHS_mid']:.5f} mmol/g"
                f"\n    Q0.10–Q0.90 = "
                f"{row['UHS_low']:.5f}–{row['UHS_high']:.5f} mmol/g"
                f"\n    Excess/pore-volume contribution = {ratio:.3f}"
            )

        print(
            "\nDepth profiles use the same training-only TabPFN model "
            "evaluated on the random test set. Quantile bands retain "
            "the raw model outputs and hold geological inputs fixed."
        )

    # ── 5c. Monte Carlo over the full input space ────────────────
    print(f"\n  Monte Carlo: {MC_N} rows, {MC_DESIGN} design, ranges widened by "
          f"±{MC_RANGE_EXPAND:.0%}")

    rng = np.random.default_rng(SEED)
    lo_obs = X.min(axis=0).astype(float)
    hi_obs = X.max(axis=0).astype(float)
    span   = hi_obs - lo_obs
    lo = lo_obs - MC_RANGE_EXPAND * span
    hi = hi_obs + MC_RANGE_EXPAND * span

    # physical floors: nothing negative, pressure and composition bounded
    lo = np.maximum(lo, 0.0)
    for j, fname in enumerate(feature_names):
        if fname in comp_cols:
            hi[j] = min(hi[j], 100.0)

    # unit design in [0,1]^d, then mapped onto the widened ranges
    d = X.shape[1]
    if MC_DESIGN == "lhs":
        try:
            from scipy.stats import qmc
            U = qmc.LatinHypercube(d=d, seed=SEED).random(MC_N)
        except Exception as e:
            print(f"    Latin hypercube unavailable ({e}); using random sampling")
            U = rng.random((MC_N, d))
    else:
        U = rng.random((MC_N, d))

    X_mc = lo + U * (hi - lo)

    # renormalise proximate analysis so the four fractions sum to 100 wt%
    if MC_RENORMALISE and len(comp_cols) == 4:
        ci = [feature_names.index(c) for c in comp_cols]
        tot = X_mc[:, ci].sum(axis=1, keepdims=True)
        tot[tot < 1e-9] = 1e-9
        X_mc[:, ci] = X_mc[:, ci] / tot * 100.0

    X_mc = X_mc.astype(np.float32)

    # how far outside the measured envelope has the sampling gone?
    outside = ((X_mc < lo_obs) | (X_mc > hi_obs)).any(axis=1).mean()
    print(f"    {outside*100:.1f}% of Monte Carlo rows fall outside the measured "
          f"range of at least one input.")
    print("    Those rows are extrapolation. Read them against the RUN 3 penalties")
    print("    before quoting the tails of the distribution.")

    # predict with the chosen model, trained on everything
    mc_model, mc_scaler = fit_model(MC_MODEL, X, y,
                                    groups_tr=groups_sample, grouped=True)
    y_mc = predict_model(MC_MODEL, mc_model, mc_scaler, X_mc)
    y_mc = np.clip(y_mc, 0.0, None)          # uptake cannot be negative

    df_mc = pd.DataFrame(X_mc, columns=feature_names)
    df_mc["H2_pred_mmol_g"] = y_mc
    df_mc["outside_measured_range"] = ((X_mc < lo_obs) | (X_mc > hi_obs)).any(axis=1)
    df_mc.to_csv(os.path.join(OUT_DIR, "run5_monte_carlo_samples.csv"), index=False)

    # percentile bands against pressure
    if i_press is not None:
        P_mc = X_mc[:, i_press]
        edges = np.linspace(P_mc.min(), P_mc.max(), 25)
        centres, p10, p50, p90, counts = [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (P_mc >= a) & (P_mc < b)
            if sel.sum() < 10:
                continue
            centres.append(0.5 * (a + b))
            p10.append(np.percentile(y_mc[sel], 10))
            p50.append(np.percentile(y_mc[sel], 50))
            p90.append(np.percentile(y_mc[sel], 90))
            counts.append(int(sel.sum()))

        df_band = pd.DataFrame(dict(pressure_MPa=centres, n=counts,
                                    P10=p10, P50=p50, P90=p90))
        df_band.to_csv(os.path.join(OUT_DIR, "run5_monte_carlo_bands.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        ax = axes[0]
        ax.scatter(P_mc, y_mc, s=5, alpha=0.18, color="#2E86AB",
                   label=f"Monte Carlo ({MC_N} realisations)")
        ax.plot(centres, p50, color="k", ls="--", lw=2.2, label="P50")
        ax.plot(centres, p10, color="#D32F2F", lw=2.2, label="P10")
        ax.plot(centres, p90, color="#D32F2F", lw=2.2, label="P90")
        ax.scatter(X[:, i_press], y, s=10, color="k", alpha=0.35,
                   zorder=5, label="measured")
        ax.set_xlabel("Pressure (MPa)")
        ax.set_ylabel("H$_2$ adsorption (mmol/g)")
        ax.set_title(f"{MC_MODEL} predictions over the sampled input space")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1]
        for P_target in MC_PRESSURES:
            tol = max(0.4, 0.03 * (P_mc.max() - P_mc.min()))
            sel = np.abs(P_mc - P_target) < tol
            if sel.sum() < 20:
                continue
            vals = np.sort(y_mc[sel])
            ax.plot(vals, np.linspace(0, 100, len(vals)), lw=2.0,
                    label=f"P = {P_target:g} MPa (n={sel.sum()})")
        ax.set_xlabel("H$_2$ adsorption (mmol/g)")
        ax.set_ylabel("Cumulative probability (%)")
        ax.set_title("Cumulative distribution at selected pressures")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.suptitle(f"Monte Carlo sensitivity — inputs sampled over the measured "
                     f"range widened by ±{MC_RANGE_EXPAND:.0%}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "run5_monte_carlo.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # percentile table at the pressures used in the manuscript
        pct_rows = []
        for P_target in MC_PRESSURES:
            tol = max(0.4, 0.03 * (P_mc.max() - P_mc.min()))
            sel = np.abs(P_mc - P_target) < tol
            if sel.sum() < 20:
                continue
            pct_rows.append(dict(
                pressure_MPa=P_target, n=int(sel.sum()),
                P10=float(np.percentile(y_mc[sel], 10)),
                P50=float(np.percentile(y_mc[sel], 50)),
                P90=float(np.percentile(y_mc[sel], 90)),
                mean=float(y_mc[sel].mean())))
        if pct_rows:
            df_pct = pd.DataFrame(pct_rows)
            df_pct.to_csv(os.path.join(OUT_DIR, "run5_monte_carlo_percentiles.csv"),
                          index=False)
            print("\n" + df_pct.round(4).to_string(index=False))

        print("\n  Note the difference between this and 5a. The Monte Carlo band is")
        print("  the spread of predictions across different coals — how variable the")
        print("  resource is. The 5a intervals are how uncertain the model is about")
        print("  one coal. Both belong in the paper, labelled distinctly.")

    # ── 5d. distribution of measured vs Monte Carlo uptake ───────
    from scipy import stats as _st

    mu_o, sd_o = float(np.mean(y)), float(np.std(y, ddof=1))
    mu_m, sd_m = float(np.mean(y_mc)), float(np.std(y_mc, ddof=1))
    sk_o, sk_m = float(_st.skew(y)), float(_st.skew(y_mc))
    ks_stat, ks_p = _st.ks_2samp(y, y_mc)

    dist_rows = [
        dict(dataset="Measured", n=len(y), mean=mu_o, sd=sd_o,
             skewness=sk_o, p10=float(np.percentile(y, 10)),
             median=float(np.median(y)), p90=float(np.percentile(y, 90)),
             min=float(y.min()), max=float(y.max()),
             shapiro_p=float(_st.shapiro(y[:5000])[1])),
        dict(dataset="Monte Carlo", n=len(y_mc), mean=mu_m, sd=sd_m,
             skewness=sk_m, p10=float(np.percentile(y_mc, 10)),
             median=float(np.median(y_mc)), p90=float(np.percentile(y_mc, 90)),
             min=float(y_mc.min()), max=float(y_mc.max()),
             shapiro_p=float(_st.shapiro(y_mc[:5000])[1])),
    ]
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(os.path.join(OUT_DIR, "run5_distribution_stats.csv"), index=False)
    print("\n  Distribution comparison, measured vs Monte Carlo:")
    print(df_dist.round(4).to_string(index=False))
    print(f"    Two-sample KS: D = {ks_stat:.3f}, p = {ks_p:.3g}")

    grid = np.linspace(min(y.min(), y_mc.min()),
                       max(y.max(), y_mc.max()), 400)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) histograms with fitted normal curves
    ax = axes[0]
    ax.hist(y, bins=35, density=True, alpha=0.45, color="#444444",
            edgecolor="k", linewidth=0.4, label=f"Measured (n={len(y)})")
    ax.hist(y_mc, bins=35, density=True, alpha=0.40, color="#2E86AB",
            edgecolor="k", linewidth=0.4, label=f"Monte Carlo (n={len(y_mc)})")
    ax.plot(grid, _st.norm.pdf(grid, mu_o, sd_o), color="k", lw=2.2,
            label=f"Normal fit, measured (μ={mu_o:.3f}, σ={sd_o:.3f})")
    ax.plot(grid, _st.norm.pdf(grid, mu_m, sd_m), color="#2E86AB", lw=2.2,
            label=f"Normal fit, Monte Carlo (μ={mu_m:.3f}, σ={sd_m:.3f})")
    ax.set_xlabel("H$_2$ adsorption (mmol/g)")
    ax.set_ylabel("Probability density")
    ax.set_title("Frequency distribution and fitted normal curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) empirical densities and cumulative comparison
    ax = axes[1]
    try:
        ax.plot(grid, _st.gaussian_kde(y)(grid), color="k", lw=2.2,
                label="Measured, kernel density")
        ax.plot(grid, _st.gaussian_kde(y_mc)(grid), color="#2E86AB", lw=2.2,
                label="Monte Carlo, kernel density")
    except Exception:
        pass
    ax.set_xlabel("H$_2$ adsorption (mmol/g)")
    ax.set_ylabel("Probability density")
    ax.set_title("Kernel densities — the shape a normal fit misses")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Measured H$_2$ adsorption against the Monte Carlo population",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "run5_distribution_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    if min(sk_o, sk_m) > 0.5:
        print("\n    Both distributions are right-skewed, so the normal curves in")
        print("    panel (a) are a poor description of either one. Show them if the")
        print("    point is to compare location and spread, but describe the data")
        print("    with the percentiles rather than with μ and σ, and keep the")
        print("    kernel densities in the figure so the shape is visible.")
    print("\n    The Monte Carlo population is a uniform sweep of the input space,")
    print("    while the measured data reflect which coals and pressures people")
    print("    chose to study. A shift between the two is expected and is not")
    print("    evidence that either is wrong — it quantifies sampling bias in the")
    print("    literature, which is a fair thing to say in the discussion.")


# ═════════════════════════════════════════════════════════════════
#  RUN 6 — relevancy factor under the isotherm-holdout protocol
# ═════════════════════════════════════════════════════════════════
# The relevancy factor of input k is the correlation between that input and the
# prediction, taken across the dataset and keeping the sign of the effect:
#
#     r_k = SUM (x_ki - xbar_k)(yhat_i - yhatbar)
#           / sqrt( SUM (x_ki - xbar_k)^2 * SUM (yhat_i - yhatbar)^2 )
#
# It ranges from -1 to +1 and describes the LINEAR component of the dependence
# only. Computing it on in-sample predictions tells you about the fit; computing
# it on out-of-fold predictions, as here, tells you about what the model has
# actually generalised. The measured-value column is the reference: if the model
# has learned the right directions, the two agree in sign and roughly in rank.

RELEVANCY_PROTOCOL = "B_holdout_isotherm"
RELEVANCY_MODEL    = "TabPFN"

if RUN_6_RELEVANCY:
    print("\n" + "="*60)
    print(f"RUN 6 — relevancy factor ({RELEVANCY_MODEL}, {RELEVANCY_PROTOCOL})")
    print("="*60)

    def relevancy_factor(Xa, ya):
        """Signed correlation of each column of Xa with ya."""
        Xa = np.asarray(Xa, dtype=float); ya = np.asarray(ya, dtype=float)
        dy = ya - ya.mean()
        out = []
        for k in range(Xa.shape[1]):
            dx = Xa[:, k] - Xa[:, k].mean()
            den = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
            out.append(float(np.sum(dx * dy) / den) if den > 1e-12 else np.nan)
        return np.array(out)

    # reuse RUN 1 out-of-fold predictions if available, otherwise recompute
    oof = None
    if protocol_preds:
        allp = pd.concat(protocol_preds, ignore_index=True)
        sel = allp[allp.protocol == RELEVANCY_PROTOCOL]
        if len(sel) and RELEVANCY_MODEL in sel.columns:
            oof = sel[["row_idx", "fold", "actual", RELEVANCY_MODEL]].copy()
            oof = oof.rename(columns={RELEVANCY_MODEL: "pred"})

    if oof is None:
        print("  RUN 1 predictions unavailable — recomputing the folds.")
        if "isotherm_id" not in df_raw.columns:
            raise RuntimeError("isotherm_id needed for the relevancy protocol")
        gk = GroupKFold(n_splits=N_FOLDS)
        parts = []
        for fold, (tr, te) in enumerate(
                gk.split(X, y, df_raw["isotherm_id"].values), 1):
            p, _, _ = fit_predict(RELEVANCY_MODEL, X[tr], y[tr], X[te],
                                  groups_tr=groups_sample[tr], grouped=True)
            parts.append(pd.DataFrame({"row_idx": te, "fold": fold,
                                       "actual": y[te], "pred": p}))
        oof = pd.concat(parts, ignore_index=True)

    idx_oof = oof["row_idx"].values.astype(int)
    X_oof   = X[idx_oof]

    rf_pred = relevancy_factor(X_oof, oof["pred"].values)
    rf_meas = relevancy_factor(X_oof, oof["actual"].values)

    # per-fold spread, so the bars can carry error bars
    per_fold = []
    for f, g in oof.groupby("fold"):
        ii = g["row_idx"].values.astype(int)
        if len(ii) < 10:
            continue
        per_fold.append(relevancy_factor(X[ii], g["pred"].values))
    per_fold = np.array(per_fold) if per_fold else np.empty((0, X.shape[1]))
    rf_sd = per_fold.std(axis=0) if len(per_fold) > 1 else np.zeros(X.shape[1])

    df_rf = pd.DataFrame({
        "feature": feature_names,
        "relevancy_factor_predicted": rf_pred,
        "relevancy_factor_measured": rf_meas,
        "sd_across_folds": rf_sd,
        "abs_rank": pd.Series(np.abs(rf_pred)).rank(ascending=False).astype(int),
        "sign_agrees_with_measured": np.sign(rf_pred) == np.sign(rf_meas),
    }).sort_values("abs_rank")
    df_rf.to_csv(os.path.join(OUT_DIR, "run6_relevancy_factor.csv"), index=False)
    print(df_rf.round(4).to_string(index=False))

    order = np.argsort(np.abs(rf_pred))          # smallest first, for the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    names = [feature_names[i] for i in order]
    vals  = rf_pred[order]
    errs  = rf_sd[order]
    colors = ["#2E86AB" if v >= 0 else "#E63946" for v in vals]
    ax.barh(names, vals, xerr=errs, capsize=3, color=colors, edgecolor="k")
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("Relevancy factor")
    ax.set_title(f"{RELEVANCY_MODEL}, out-of-fold predictions\n"
                 f"({RELEVANCY_PROTOCOL.replace('_', ' ')})")
    ax.grid(alpha=0.3, axis="x")
    lim = float(np.nanmax(np.abs(rf_pred)) + np.nanmax(errs) + 0.08)
    ax.set_xlim(-lim, lim)
    for yv, v in zip(range(len(vals)), vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yv, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)

    ax = axes[1]
    w = 0.38
    pos = np.arange(len(feature_names))
    ax.barh(pos - w/2, rf_pred, height=w, color="#2E86AB",
            edgecolor="k", label=f"{RELEVANCY_MODEL} prediction")
    ax.barh(pos + w/2, rf_meas, height=w, color="#999999",
            edgecolor="k", label="measured data")
    ax.set_yticks(pos); ax.set_yticklabels(feature_names, fontsize=9)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("Relevancy factor")
    ax.set_title("Model against data — do the directions agree?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")

    plt.suptitle("Input contribution to predicted H$_2$ adsorption",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "run6_relevancy_factor.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    n_disagree = int((~df_rf.sign_agrees_with_measured).sum())
    if n_disagree:
        bad = df_rf.loc[~df_rf.sign_agrees_with_measured, "feature"].tolist()
        print(f"\n  {n_disagree} feature(s) where the model's direction disagrees")
        print(f"  with the measured data: {bad}")
        print("  Check these before drawing mechanistic conclusions from the ranking.")
    print("\n  The ranking reflects the ranges sampled in the source experiments as")
    print("  much as the underlying physics. Pressure spans 0-18 MPa here while")
    print("  temperature spans about 90 K, so their relative order is partly an")
    print("  artefact of the experimental design. Say so when you report it.")


# ═════════════════════════════════════════════════════════════════
#  Consolidated export
# ═════════════════════════════════════════════════════════════════

xlsx_path = os.path.join(OUT_DIR, "paper_tables.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
    meta = pd.DataFrame([
        dict(item="measurements", value=len(df_raw)),
        dict(item="coal samples", value=n_samples),
        dict(item="sources", value=(len(np.unique(source_id)) if source_id is not None else "not labelled")),
        dict(item="features", value=len(feature_names)),
        dict(item="feature names", value=", ".join(feature_names)),
        dict(item="composition columns used for grouping", value=", ".join(map(str, comp_cols))),
    ])
    meta.to_excel(xl, sheet_name="Dataset", index=False)

    for fname, sheet in [
        ("run1_protocol_metrics.csv",   "Run1_protocols"),
        ("run2_data_efficiency.csv",    "Run2_efficiency"),
        ("run3_extrapolation.csv",      "Run3_extrapolation"),
        ("run4_physics_violations.csv", "Run4_physics"),
        ("run5_interval_coverage.csv",  "Run5_coverage"),
        ("run5_depth_capacity.csv",     "Run5_depth"),
        ("run5_monte_carlo_bands.csv",       "Run5_MC_bands"),
        ("run5_monte_carlo_percentiles.csv", "Run5_MC_percentiles"),
        ("run3_temperature_monotonicity.csv","Run3_T_monotonicity"),
        ("run1_single_random_split.csv",     "Run1_single_split"),
        ("run4b_vant_hoff_summary.csv",      "Run4b_vantHoff"),
        ("run5_distribution_stats.csv",      "Run5_distributions"),
        ("run6_relevancy_factor.csv",        "Run6_relevancy"),
    ]:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            pd.read_csv(p).to_excel(xl, sheet_name=sheet, index=False)

print(f"\n{'='*60}")
print(f"Done. Tables and figures in {OUT_DIR}")
print(f"Consolidated workbook: {xlsx_path}")
print("="*60)


# ─────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────
def predict_h2_ads(*features, model_path=os.path.join(OUT_DIR, "tabpfn_h2_ads_model.pkl")):
    """Predict H2 adsorption (mmol/g) for one sample, in the training column order."""
    bundle = joblib.load(model_path)
    s = np.array([list(features)], dtype=np.float32)
    s = bundle["scaler_X"].transform(s).astype(np.float32)
    return float(tabpfn_predict(bundle["model"], s,
                                bundle["config"].get("output_type", "mean"))[0])
