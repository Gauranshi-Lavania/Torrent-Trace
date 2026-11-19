# RAM RAM 
#RADHE RADHE 


#FLOOD PREDICTION 
#!/usr/bin/env python3
"""
train.py - Training pipeline for the Flood Risk Studio project
==============================================================

Kaggle: Playground Series S4E5 - "Regression with a Flood Prediction Dataset"

What this script does (in order)
--------------------------------
1. Unzips ``flood-prediction.zip`` (train.csv + test.csv) into ``data/``.
2. Loads train.csv and shrinks memory (int64 -> int8, float64 -> float32).
3. Splits the data into train / validation / test (70 / 15 / 15).
4. Runs a feature-engineering ABLATION study (Ridge + LightGBM):
   raw 20 factors  ->  + sum/mean  ->  + std  ->  + min/max  ->  + skew.
5. Trains Linear Regression, Ridge, LightGBM, XGBoost and CatBoost on the
   full engineered feature set.
6. Learns weighted-ensemble weights on the validation split (SLSQP, w >= 0, sum w = 1).
7. Reports R2 / RMSE / MAE on the untouched TEST split.
8. Saves everything the web app needs into ``saved_models/`` via joblib/JSON.

Usage
-----
    python train.py                    # full run on all rows (several minutes)
    python train.py --fast             # fewer trees, quick sanity check
    python train.py --sample 200000    # use a random subset of rows
    python train.py --zip path/to/flood-prediction.zip

This file is also imported by ``app.py`` so that training and serving use
exactly the same ``engineer_features`` function (no train/serve skew).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize
from scipy.stats import skew
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

log = logging.getLogger("flood.train")

# --------------------------------------------------------------------------- #
# Constants shared with app.py
# --------------------------------------------------------------------------- #
TARGET = "FloodProbability"
ID_COL = "id"

# Used only to group sliders in the UI. The real feature list is auto-detected
# from train.csv and stored in metadata.json, so a spelling difference in a
# column name can never break the pipeline.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Natural factors": [
        "MonsoonIntensity", "TopographyDrainage", "ClimateChange",
        "Landslides", "Watersheds", "CoastalVulnerability",
    ],
    "Land use & human activity": [
        "Deforestation", "Urbanization", "AgriculturalPractices",
        "Encroachments", "WetlandsLoss",
    ],
    "Infrastructure & management": [
        "RiverManagement", "DamsQuality", "Siltation", "DrainageSystems",
        "DeterioratingInfrastructure", "InadequatePlanning",
        "IneffectiveDisasterPreparedness",
    ],
    "Social & political": ["PopulationScore", "PoliticalFactors"],
}

ALL_STATS = ["RowSum", "RowMean", "RowStd", "RowMax", "RowMin", "RowSkew"]

# Cumulative stages for the ablation study: (label, statistics added so far)
ABLATION_STAGES: List[Tuple[str, List[str]]] = [
    ("Raw 20 factors", []),
    ("+ Sum & Mean", ["RowSum", "RowMean"]),
    ("+ Std", ["RowSum", "RowMean", "RowStd"]),
    ("+ Min & Max", ["RowSum", "RowMean", "RowStd", "RowMax", "RowMin"]),
    ("+ Skew (full set)", ALL_STATS),
]

# File stems used when saving to disk (app.py reads the same names)
MODEL_FILES = {
    "Linear Regression": "linear_regression",
    "Ridge Regression": "ridge",
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
}
ENSEMBLE_NAME = "Weighted Ensemble"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def ensure_extracted(zip_path: Path, data_dir: Path) -> Path:
    """Unzip the Kaggle archive (if needed) and return the path to train.csv.

    Skips extraction when a train.csv already exists in ``data_dir``.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(data_dir.rglob("train.csv"))
    if existing:
        log.info("Found existing %s - skipping unzip.", existing[0])
        return existing[0]

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find '{zip_path}'. Place flood-prediction.zip next to "
            "train.py or pass its location with --zip."
        )
    try:
        with zipfile.ZipFile(zip_path) as zf:
            log.info("Extracting %s -> %s", zip_path, data_dir)
            zf.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"'{zip_path}' is not a valid zip file.") from exc

    found = list(data_dir.rglob("train.csv"))
    if not found:
        raise FileNotFoundError("The archive did not contain a train.csv file.")
    return found[0]


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns (int64 -> int8/16/32, float64 -> float32)."""
    before = df.memory_usage(deep=True).sum() / 1e6
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    after = df.memory_usage(deep=True).sum() / 1e6
    log.info("Memory: %.1f MB -> %.1f MB (%.0f%% smaller)", before, after, 100 * (1 - after / before))
    return df


def load_train(csv_path: Path, sample: Optional[int], seed: int) -> Tuple[pd.DataFrame, List[str], int]:
    """Read train.csv, optimise memory and detect the feature columns.

    Returns (dataframe, feature_names, total_rows_in_file).
    """
    log.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")

    features = [c for c in df.columns if c not in (ID_COL, TARGET)]
    if df[features + [TARGET]].isna().any().any():
        n_bad = int(df[features + [TARGET]].isna().any(axis=1).sum())
        log.warning("Dropping %d rows with missing values.", n_bad)
        df = df.dropna(subset=features + [TARGET])

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
        log.info("Using a random sample of %d rows.", sample)

    df = df[features + [TARGET]].reset_index(drop=True)
    df = reduce_memory(df)
    log.info("Detected %d feature columns.", len(features))
    return df, features, total_rows


# --------------------------------------------------------------------------- #
# Feature engineering (shared with app.py)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, features: Sequence[str],
                      stats: Sequence[str] = tuple(ALL_STATS)) -> pd.DataFrame:
    """Return the raw factors plus row-wise statistics across those factors.

    Parameters
    ----------
    df : DataFrame containing at least the ``features`` columns.
    features : the 20 raw factor names, in training order.
    stats : which statistics to add (subset of ``ALL_STATS``).

    Notes
    -----
    Values are cast to float32 *before* aggregating: summing int8 columns
    would overflow (int8 max is 127).
    """
    features = list(features)
    out = df[features].astype(np.float32).copy()
    x = out.to_numpy(dtype=np.float32)
    new_cols: Dict[str, np.ndarray] = {}
    if "RowSum" in stats:
        new_cols["RowSum"] = x.sum(axis=1)
    if "RowMean" in stats:
        new_cols["RowMean"] = x.mean(axis=1)
    if "RowStd" in stats:
        new_cols["RowStd"] = x.std(axis=1)
    if "RowMax" in stats:
        new_cols["RowMax"] = x.max(axis=1)
    if "RowMin" in stats:
        new_cols["RowMin"] = x.min(axis=1)
    if "RowSkew" in stats:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            # A row with identical values has zero variance -> skew is NaN -> use 0.
            new_cols["RowSkew"] = np.nan_to_num(skew(x, axis=1), nan=0.0)
    for name in ALL_STATS:  # keep a fixed column order
        if name in new_cols:
            out[name] = new_cols[name].astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Metrics & model factories
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """R2, RMSE and MAE (predictions are clipped to the valid [0, 1] range)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def lgbm_params(seed: int, n_jobs: int, fast: bool, lr: float = 0.03, trees: int = 3000) -> dict:
    return dict(
        n_estimators=300 if fast else trees, learning_rate=0.1 if fast else lr,
        num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
    )


def fit_model(name: str, model, X_tr, y_tr, X_va, y_va):
    """Fit one model; boosters use the validation split for early stopping."""
    if name == "LightGBM":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    elif name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "CatBoost":
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    else:  # sklearn linear models
        model.fit(X_tr, y_tr)
    return model


def build_models(seed: int, n_jobs: int, fast: bool) -> Dict[str, object]:
    """Create the five base models with sensible, fairly conservative settings."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "LightGBM": LGBMRegressor(**lgbm_params(seed, n_jobs, fast)),
        "XGBoost": XGBRegressor(
            n_estimators=300 if fast else 3000, learning_rate=0.1 if fast else 0.03,
            max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="rmse", early_stopping_rounds=100, random_state=seed, n_jobs=n_jobs,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300 if fast else 2000, learning_rate=0.1 if fast else 0.06,
            depth=6, loss_function="RMSE", early_stopping_rounds=150,
            random_seed=seed, thread_count=n_jobs, verbose=0,
        ),
    }


# --------------------------------------------------------------------------- #
# Ablation study
# --------------------------------------------------------------------------- #
def run_ablation(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                 features: List[str], seed: int, n_jobs: int, fast: bool):
    """Compare raw vs. engineered feature sets with Ridge and LightGBM.

    Returns (ablation_rows, raw_lightgbm_model). The raw LightGBM model is kept
    as the SHAP explainer because its inputs are the 20 human-readable factors.
    """
    rows, raw_lgbm = [], None
    base_r2 = {}
    for i, (label, stats) in enumerate(ABLATION_STAGES):
        t0 = time.time()
        X_tr, X_va, X_te = (engineer_features(d, features, stats) for d in (tr, va, te))
        y_tr, y_va, y_te = tr[TARGET].values, va[TARGET].values, te[TARGET].values

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
        m_ridge = evaluate(y_te, ridge.predict(X_te))

        lgbm = LGBMRegressor(**lgbm_params(seed, n_jobs, fast=True, lr=0.08, trees=800))
        fit_model("LightGBM", lgbm, X_tr, y_tr, X_va, y_va)
        m_lgbm = evaluate(y_te, lgbm.predict(X_te))

        if i == 0:
            raw_lgbm = lgbm
            base_r2 = {"ridge": m_ridge["R2"], "lgbm": m_lgbm["R2"]}
        rows.append({
            "Feature set": label,
            "# features": X_tr.shape[1],
            "Ridge R2": m_ridge["R2"], "Ridge RMSE": m_ridge["RMSE"],
            "LightGBM R2": m_lgbm["R2"], "LightGBM RMSE": m_lgbm["RMSE"],
            "Ridge dR2 vs raw": m_ridge["R2"] - base_r2["ridge"],
            "LightGBM dR2 vs raw": m_lgbm["R2"] - base_r2["lgbm"],
        })
        log.info("Ablation [%-18s] Ridge R2=%.5f | LightGBM R2=%.5f (%.1fs)",
                 label, m_ridge["R2"], m_lgbm["R2"], time.time() - t0)
    return rows, raw_lgbm


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def optimise_weights(pred_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 that minimise validation MSE."""
    m = pred_matrix.shape[1]
    objective = lambda w: float(np.mean((pred_matrix @ w - y) ** 2))
    res = minimize(
        objective, x0=np.full(m, 1.0 / m), method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
    )
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train flood-risk models and save artifacts.")
    p.add_argument("--zip", type=Path, default=None, help="Path to flood-prediction.zip")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("saved_models"))
    p.add_argument("--sample", type=int, default=None, help="Use only N random rows")
    p.add_argument("--fast", action="store_true", help="Fewer trees (quick test run)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.out_dir / "train.log", mode="w", encoding="utf-8")],
    )
    t_start = time.time()
    np.random.seed(args.seed)

    # ---- 1. Locate, unzip and load data -----------------------------------
    zip_candidates = [args.zip] if args.zip else [Path("flood-prediction.zip"), Path("data/flood-prediction.zip")]
    zip_path = next((z for z in zip_candidates if z and z.exists()), zip_candidates[0])
    try:
        csv_path = ensure_extracted(zip_path, args.data_dir)
        df, features, total_rows = load_train(csv_path, args.sample, args.seed)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    # ---- 2. Split: 70% train / 15% validation / 15% test -------------------
    tmp, te = train_test_split(df, test_size=0.15, random_state=args.seed)
    tr, va = train_test_split(tmp, test_size=0.15 / 0.85, random_state=args.seed)
    log.info("Split sizes -> train: %d | valid: %d | test: %d", len(tr), len(va), len(te))

    # ---- 3. Ablation study --------------------------------------------------
    log.info("Running feature-engineering ablation study ...")
    ablation_rows, explainer_model = run_ablation(tr, va, te, features, args.seed, args.n_jobs, args.fast)

    # ---- 4. Train all base models on the full engineered feature set -------
    X_tr, X_va, X_te = (engineer_features(d, features) for d in (tr, va, te))
    y_tr, y_va, y_te = tr[TARGET].values, va[TARGET].values, te[TARGET].values
    log.info("Engineered feature matrix: %d columns", X_tr.shape[1])

    models = build_models(args.seed, args.n_jobs, args.fast)
    val_preds, test_preds, metrics = {}, {}, []
    for name, model in models.items():
        t0 = time.time()
        try:
            fit_model(name, model, X_tr, y_tr, X_va, y_va)
        except Exception as exc:  # keep going if one library misbehaves
            log.exception("Training %s failed: %s", name, exc)
            return 2
        val_preds[name] = np.clip(model.predict(X_va), 0, 1)
        test_preds[name] = np.clip(model.predict(X_te), 0, 1)
        m = evaluate(y_te, test_preds[name])
        metrics.append({"Model": name, **m})
        log.info("%-18s R2=%.5f RMSE=%.5f MAE=%.5f (%.1fs)", name, m["R2"], m["RMSE"], m["MAE"], time.time() - t0)
        joblib.dump(model, args.out_dir / f"{MODEL_FILES[name]}.joblib", compress=3)

    # ---- 5. Weighted ensemble ----------------------------------------------
    names = list(models)
    weights = optimise_weights(np.column_stack([val_preds[n] for n in names]), y_va)
    ens_test = np.column_stack([test_preds[n] for n in names]) @ weights
    m = evaluate(y_te, ens_test)
    metrics.append({"Model": ENSEMBLE_NAME, **m})
    weight_dict = {n: float(w) for n, w in zip(names, weights)}
    log.info("%-18s R2=%.5f RMSE=%.5f MAE=%.5f", ENSEMBLE_NAME, m["R2"], m["RMSE"], m["MAE"])
    log.info("Ensemble weights: %s", {k: round(v, 3) for k, v in weight_dict.items()})
    joblib.dump({"weights": weight_dict}, args.out_dir / "ensemble.joblib")

    # ---- 6. Artifacts for the web app ---------------------------------------
    joblib.dump(explainer_model, args.out_dir / "explainer_lgbm_raw.joblib", compress=3)
    df.sample(n=min(500, len(df)), random_state=args.seed).to_csv(args.out_dir / "background_sample.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(args.out_dir / "ablation.csv", index=False)

    y_all = df[TARGET].to_numpy(dtype=np.float64)
    q = df[features].quantile([0.25, 0.5, 0.75])
    feature_stats = {
        f: {"min": int(df[f].min()), "max": int(df[f].max()), "mean": float(df[f].mean()),
            "q25": float(q.loc[0.25, f]), "q50": float(q.loc[0.5, f]), "q75": float(q.loc[0.75, f])}
        for f in features
    }
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "engineered_stats": ALL_STATS,
        "dataset": {"rows_in_file": total_rows, "rows_used": len(df),
                    "train": len(tr), "valid": len(va), "test": len(te)},
        "config": {"seed": args.seed, "fast": args.fast, "sample": args.sample},
        "metrics": metrics,
        "ensemble_weights": weight_dict,
        "ablation": ablation_rows,
        "feature_stats": feature_stats,
        "target_stats": {"mean": float(y_all.mean()), "std": float(y_all.std()),
                         "min": float(y_all.min()), "max": float(y_all.max())},
        "target_quantiles": np.quantile(y_all, np.linspace(0, 1, 21)).tolist(),
        # Risk tiers are calibrated to the data: the bottom third of regions is
        # "Low", the top third is "High" (labels are relative, see README).
        "tiers": {"low_max": float(np.quantile(y_all, 1 / 3)), "high_min": float(np.quantile(y_all, 2 / 3))},
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

    log.info("Done in %.1f min. Artifacts saved to '%s/'. Launch the UI with: streamlit run app.py",
             (time.time() - t_start) / 60, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())