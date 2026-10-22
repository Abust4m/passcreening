from __future__ import annotations
import argparse
import json
import os
import re
import time
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

DEFAULT_MORGAN_BITS = 1024
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_DFT_COLS = ["HOMO", "Gap", "DM", "is_ionic"]
N_JOBS = 1
INNER_SPLITS = 3
DESC_NAMES = [name for name, _ in Descriptors._descList]
FORBIDDEN_FEATURES = {
    "deltaPCE", "dPCE", "labelln", "class1.8", "class1.9", "class2.0",
    "PCEclass", "report PCE", "Initial PCE", "after PCE",
}
FAMILY_ORDER = {"lr": 0, "rf": 1, "svm": 2, "mlp": 3, "xgb": 4, "gpc": 5}
STACK_BASES = ["rf", "svm", "xgb", "gpc"]

DEFAULTS = {
    "lr": {"C": 0.1},
    "rf": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 1, "max_features": 0.2},
    "svm": {"C": 1.0, "gamma": 0.1},
    "mlp": {"h1": 100, "h2": 50, "alpha": 0.001, "lr": 0.001},
    "xgb": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0,
            "reg_alpha": 0.0, "reg_lambda": 1.0},
    "gpc": {"kernel": "rbf"},
}

def norm_smiles(smiles):
    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol) if mol is not None else re.sub(r"\s+", "", text)


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_sample_weights(delta_pce, threshold, buffer_width=0.1, min_weight=0.3):
    distance = np.abs(np.asarray(delta_pce, dtype=float) - threshold)
    sigma = buffer_width / 2
    normalization = 1 - np.exp(-(buffer_width ** 2) / (2 * sigma ** 2))
    return np.where(
        distance <= buffer_width,
        min_weight + (1 - min_weight) * (1 - np.exp(-(distance ** 2) / (2 * sigma ** 2))) / normalization,
        1.0,
    )


def bootstrap_ci(y_true, y_prob, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        auc = safe_auc(y_true[idx], y_prob[idx])
        if not np.isnan(auc):
            values.append(auc)
    if not values:
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def featurize(data: pd.DataFrame, dft_cols, morgan_bits=DEFAULT_MORGAN_BITS,
              morgan_radius=DEFAULT_MORGAN_RADIUS):
    for column in ["smiles"] + list(dft_cols):
        if column not in data.columns:
            raise SystemExit(f"input table is missing column {column!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
    blocks_morgan, blocks_maccs, blocks_desc, keep = [], [], [], []
    for pos, smiles in enumerate(data["smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            continue
        blocks_morgan.append(np.asarray(generator.GetFingerprint(mol), dtype=np.float64))
        blocks_maccs.append(np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float64))
        row = []
        for _, func in Descriptors._descList:
            try:
                value = float(func(mol))
            except Exception:
                value = 0.0
            row.append(value if np.isfinite(value) else 0.0)
        blocks_desc.append(np.asarray(row, dtype=np.float64))
        keep.append(pos)

    if not keep:
        raise SystemExit("No valid SMILES in the input table.")

    dft = data[dft_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)[keep]
    dft = np.nan_to_num(dft, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.hstack([np.vstack(blocks_morgan), np.vstack(blocks_maccs),
                   np.vstack(blocks_desc), dft])
    columns = ([f"morgan_{i}" for i in range(morgan_bits)]
               + [f"maccs_{i}" for i in range(167)]
               + list(DESC_NAMES)
               + list(dft_cols))
    X_df = pd.DataFrame(X, columns=columns)
    data_valid = data.iloc[keep].reset_index(drop=True)

    overlap = set(X_df.columns) & FORBIDDEN_FEATURES
    assert not overlap, f"label / PCE columns leaked into features: {sorted(overlap)}"
    return X_df, data_valid


def lasso_mask(Z, y, seed, cs):

    n_features = Z.shape[1]
    if n_features == 0:
        return np.zeros(0, dtype=bool)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.ones(n_features, dtype=bool)

    cs = tuple(float(c) for c in cs) or (0.1,)
    if len(cs) > 1:
        n_min = int(np.bincount(y).min())
        cv = max(2, min(5, n_min))
        selector = LogisticRegressionCV(Cs=list(cs), cv=cv, penalty="l1", solver="saga",
                                        scoring="roc_auc", max_iter=5000,
                                        random_state=seed)
    else:
        selector = LogisticRegression(C=cs[0], penalty="l1", solver="liblinear",
                                      max_iter=5000, random_state=seed)
    selector.fit(Z, y)
    coef = np.abs(np.ravel(selector.coef_))
    positive = coef[coef > 0]
    threshold = float(np.percentile(positive, 50)) if positive.size > 50 else 0.0
    mask = coef > threshold
    if not mask.any():
        top = min(20, coef.size)
        mask = np.zeros_like(coef, dtype=bool)
        mask[np.argsort(coef)[-top:]] = True
    return mask


class LeakFreePreprocessor:

    def __init__(self, pruned_columns, scaler, mask, selected_columns, desc_columns):
        self.pruned_columns = list(pruned_columns)
        self.scaler = scaler
        self.mask = np.asarray(mask, dtype=bool)
        self.selected_columns = list(selected_columns)
        self.desc_columns = list(desc_columns)
        assert len(self.mask) == len(self.pruned_columns)

    @classmethod
    def fit(cls, X_train, y_train, desc_columns, seed, cs=(0.1,), collinear_thr=0.8):
        X = X_train.astype(float)
        constant = [c for c in X.columns if float(X[c].std(ddof=0)) == 0.0]
        desc = [c for c in desc_columns if c in X.columns and c not in constant]

        dropped = set()
        if len(desc) > 1:
            corr = X[desc].corr().abs().to_numpy()
            for i, first in enumerate(desc):
                if first in dropped:
                    continue
                for j in range(i + 1, len(desc)):
                    second = desc[j]
                    if second not in dropped and corr[i, j] > collinear_thr:
                        dropped.add(second)

        pruned = [c for c in X.columns if c not in constant and c not in dropped]
        scaler = StandardScaler().fit(X[pruned])
        mask = lasso_mask(scaler.transform(X[pruned]), np.asarray(y_train), seed, cs)
        selected = [c for c, keep in zip(pruned, mask) if keep]
        return cls(pruned, scaler, mask, selected, desc_columns)

    def transform(self, X):
        Z = self.scaler.transform(X.astype(float)[self.pruned_columns])
        return Z[:, self.mask]

    def as_dict(self):
        return {
            "pruned_columns": self.pruned_columns,
            "scaler": self.scaler,
            "mask": self.mask,
            "selected_columns": self.selected_columns,
            "desc_columns": self.desc_columns,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["pruned_columns"], payload["scaler"], payload["mask"],
                   payload["selected_columns"], payload["desc_columns"])


def get_preprocessor(cache, X_df, y, rows, seed, cs):
    key = np.asarray(rows).tobytes()
    if key not in cache:
        cache[key] = LeakFreePreprocessor.fit(X_df.iloc[rows], np.asarray(y)[rows],
                                              DESC_NAMES, seed, cs)
    return cache[key]

def sample_params(family, rng):
    if family == "lr":
        return {"C": float(10 ** rng.uniform(-2.0, 1.0))}
    if family == "rf":
        return {"n_estimators": int(rng.choice([200, 300, 500])),
                "max_depth": int(rng.integers(3, 10)),
                "min_samples_leaf": int(rng.integers(1, 4)),
                "max_features": float(rng.choice([0.1, 0.2, 0.4, 0.6]))}
    if family == "svm":
        return {"C": float(10 ** rng.uniform(-1.0, 1.3)),
                "gamma": float(10 ** rng.uniform(-3.0, 0.0))}
    if family == "mlp":
        return {"h1": int(rng.integers(32, 129)), "h2": int(rng.integers(16, 65)),
                "alpha": float(10 ** rng.uniform(-4.0, -1.5)),
                "lr": float(10 ** rng.uniform(-3.5, -2.0))}
    if family == "xgb":
        return {"n_estimators": int(rng.integers(200, 600)),
                "max_depth": int(rng.integers(2, 7)),
                "learning_rate": float(10 ** rng.uniform(-2.5, -0.7)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": float(rng.uniform(1.0, 8.0)),
                "gamma": float(rng.uniform(0.0, 3.0)),
                "reg_alpha": float(10 ** rng.uniform(-2.0, 0.0)),
                "reg_lambda": float(10 ** rng.uniform(-1.0, 1.0))}
    if family == "gpc":
        return {"kernel": str(rng.choice(["rbf", "matern15", "matern25", "rq", "const_rbf"]))}
    raise ValueError(f"unknown family: {family}")


def make_kernel(name):
    if name == "rbf":
        return RBF(length_scale=1.0)
    if name == "matern15":
        return Matern(length_scale=1.0, nu=1.5)
    if name == "matern25":
        return Matern(length_scale=1.0, nu=2.5)
    if name == "rq":
        return RationalQuadratic(length_scale=1.0, alpha=1.0)
    if name == "const_rbf":
        return ConstantKernel(1.0) * RBF(length_scale=1.0)
    raise ValueError(f"unknown kernel: {name}")


def build_model(family, params, seed):
    if family == "lr":
        return LogisticRegression(C=params["C"], max_iter=5000, random_state=seed)
    if family == "rf":
        return RandomForestClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      max_features=params["max_features"],
                                      random_state=seed, n_jobs=N_JOBS)
    if family == "svm":
        return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                   probability=True, random_state=seed)
    if family == "mlp":
        return MLPClassifier(hidden_layer_sizes=(params["h1"], params["h2"]),
                             alpha=params["alpha"], learning_rate_init=params["lr"],
                             max_iter=800, early_stopping=True, n_iter_no_change=20,
                             random_state=seed)
    if family == "xgb":
        return XGBClassifier(n_estimators=params["n_estimators"],
                             max_depth=params["max_depth"],
                             learning_rate=params["learning_rate"],
                             subsample=params["subsample"],
                             colsample_bytree=params["colsample_bytree"],
                             min_child_weight=params["min_child_weight"],
                             gamma=params["gamma"], reg_alpha=params["reg_alpha"],
                             reg_lambda=params["reg_lambda"],
                             random_state=seed, n_jobs=N_JOBS,
                             eval_metric="logloss", tree_method="hist")
    if family == "gpc":
        return GaussianProcessClassifier(kernel=make_kernel(params["kernel"]),
                                         random_state=seed)
    raise ValueError(f"unknown family: {family}")


def fit_estimator(model, X, y, sample_weight=None):
    if sample_weight is not None:
        try:
            return model.fit(X, y, sample_weight=sample_weight)
        except (TypeError, ValueError):
            pass
    return model.fit(X, y)

def make_splits(kind, y, groups, n_splits, seed):
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))
    counts = np.bincount(y) if y.size else np.array([0])
    smallest = int(counts[counts > 0].min()) if counts.size else 0

    if kind == "group" and groups is not None and StratifiedGroupKFold is not None:
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(int(n_splits), n_groups, smallest or int(n_splits)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(idx[a], idx[b]) for a, b in splitter.split(idx.reshape(-1, 1), y, groups)]

    n_splits = max(2, min(int(n_splits), smallest or 2))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(idx[a], idx[b]) for a, b in splitter.split(idx.reshape(-1, 1), y)]


def assert_disjoint(groups, train_idx, val_idx, label=""):
    shared = set(groups[train_idx]) & set(groups[val_idx])
    assert not shared, f"{label}: {len(shared)} group(s) shared between train and validation"

def tune_family(family, X_df, y, groups, train_rows, seed, n_iter, cache, weights,
                kind, cs):
    rng = np.random.default_rng(seed + 1000 * (FAMILY_ORDER.get(family, 0) + 1))
    candidates = [sample_params(family, rng) for _ in range(max(1, int(n_iter)))]
    candidates.append(DEFAULTS[family])

    inner = make_splits(kind, y[train_rows], groups[train_rows], INNER_SPLITS, seed)
    best_params, best_score = DEFAULTS[family], -np.inf
    for params in candidates:
        fold_auc = []
        for a, b in inner:
            fit_rows, val_rows = train_rows[a], train_rows[b]
            prep = get_preprocessor(cache, X_df, y, fit_rows, seed, cs)
            X_fit = prep.transform(X_df.iloc[fit_rows])
            X_val = prep.transform(X_df.iloc[val_rows])
            w = None if weights is None else weights[fit_rows]
            model = fit_estimator(build_model(family, params, seed), X_fit, y[fit_rows], w)
            fold_auc.append(safe_auc(y[val_rows], model.predict_proba(X_val)[:, 1]))
        score = float(np.nanmean(fold_auc)) if len(fold_auc) else -np.inf
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score


def fit_stack(bases, meta, X_train, y_train, groups_train, X_val, kind, seed, weights):
    n = len(y_train)
    oof = np.zeros((n, len(bases)))
    inner = make_splits(kind, y_train, groups_train, INNER_SPLITS, seed)
    for a, b in inner:
        for j, base in enumerate(bases):
            w = None if weights is None else weights[a]
            model = fit_estimator(clone(base), X_train[a], y_train[a], w)
            oof[b, j] = model.predict_proba(X_train[b])[:, 1]
    meta_model = clone(meta).fit(oof, y_train)
    val_meta = np.column_stack([
        fit_estimator(clone(base), X_train, y_train, weights).predict_proba(X_val)[:, 1]
        for base in bases
    ])
    return meta_model.predict_proba(val_meta)[:, 1]

def run_protocol(kind, X_df, y, groups, weights, families, seed, n_splits, n_iter,
                 cache, cs, verbose=True):
    splits = make_splits(kind, y, groups, n_splits, seed)
    results = defaultdict(list)
    selected_sizes = []

    for fold, (train_rows, val_rows) in enumerate(splits, 1):
        if kind == "group":
            assert_disjoint(groups, train_rows, val_rows, f"fold {fold}")

        prep = get_preprocessor(cache, X_df, y, train_rows, seed, cs)
        X_train = prep.transform(X_df.iloc[train_rows])
        X_val = prep.transform(X_df.iloc[val_rows])
        selected_sizes.append(X_train.shape[1])
        if verbose:
            print(f"  [{kind}] fold {fold}/{len(splits)}: train={len(train_rows)} "
                  f"val={len(val_rows)} selected_features={X_train.shape[1]}", flush=True)

        tuned = {}
        for family in families:
            tuned[family], _ = tune_family(family, X_df, y, groups, train_rows, seed,
                                           n_iter, cache, weights, kind, cs)

        for family in families:
            try:
                model = fit_estimator(build_model(family, tuned[family], seed), X_train,
                                      y[train_rows], weights[train_rows] if weights is not None else None)
                results[family].append((y[val_rows], model.predict_proba(X_val)[:, 1]))
            except Exception as exc:
                print(f"    [warn] {kind}/{family} fold {fold} failed: {exc}", flush=True)

        try:
            voters = [(family, build_model(family, tuned[family], seed)) for family in families]
            vote = VotingClassifier(estimators=voters, voting="soft", n_jobs=N_JOBS)
            vote.fit(X_train, y[train_rows])
            results["vote_all"].append((y[val_rows], vote.predict_proba(X_val)[:, 1]))
        except Exception as exc:
            print(f"    [warn] {kind}/vote_all fold {fold} failed: {exc}", flush=True)

        try:
            bases = [build_model(family, tuned[family], seed)
                     for family in STACK_BASES if family in tuned]
            if len(bases) >= 2:
                prob = fit_stack(bases, LogisticRegression(max_iter=2000, random_state=seed),
                                 X_train, y[train_rows], groups[train_rows], X_val, kind, seed,
                                 weights[train_rows] if weights is not None else None)
                results["stack_lr"].append((y[val_rows], prob))
        except Exception as exc:
            print(f"    [warn] {kind}/stack_lr fold {fold} failed: {exc}", flush=True)

    return results, selected_sizes


def summarize(results):
    rows = []
    for name, folds in results.items():
        aucs, accs, f1s = [], [], []
        for y_true, y_prob in folds:
            aucs.append(safe_auc(y_true, y_prob))
            pred = (np.asarray(y_prob) >= 0.5).astype(int)
            accs.append(accuracy_score(y_true, pred))
            f1s.append(f1_score(y_true, pred, zero_division=0))
        rows.append({
            "model": name,
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
            "acc_mean": float(np.mean(accs)),
            "f1_mean": float(np.mean(f1s)),
            "n_folds": len(folds),
        })
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def summarize_holdout(y_true, y_prob, seed):
    pred = (np.asarray(y_prob) >= 0.5).astype(int)
    low, high = bootstrap_ci(y_true, y_prob, seed=seed)
    return {
        "auc": safe_auc(y_true, y_prob),
        "auc_ci_low": low,
        "auc_ci_high": high,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(np.asarray(y_true) == 1)),
    }

def build_parser():
    parser = argparse.ArgumentParser(description="Leakage-free passivator classifier")
    parser.add_argument("--data", default=r"d:\demo1\demo1\train_data.csv",
                        help="modelling table (smiles, deltaPCE, class2.0, HOMO/Gap/DM/is_ionic)")
    parser.add_argument("--label", default="class2.0")
    parser.add_argument("--dft-cols", nargs="+", default=DEFAULT_DFT_COLS)
    parser.add_argument("--doi-csv", default="data_from_ref.csv",
                        help="reference table with a DOI column, used when --group-by doi")
    parser.add_argument("--group-by", choices=["smiles", "doi"], default="smiles")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=8, help="random configs per family")
    parser.add_argument("--families", nargs="+", default=list(FAMILY_ORDER),
                        choices=list(FAMILY_ORDER))
    parser.add_argument("--lasso-cs", default="0.1",
                        help="one C, or several comma separated Cs for an inner CV search")
    parser.add_argument("--weight-threshold", type=float, default=None,
                        help="deltaPCE threshold for sample weights (default: label based)")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--leak-check", dest="leak_check", action="store_true", default=True)
    parser.add_argument("--no-leak-check", dest="leak_check", action="store_false")
    parser.add_argument("--quick", action="store_true", help="fewer configs, 3 outer folds")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallelism for RF/XGB/voting (1 = safest)")
    parser.add_argument("--predict", default=None,
                        help="CSV to score with a saved run (needs --run)")
    parser.add_argument("--run", default=None, help="run directory produced by training")
    parser.add_argument("--predict-out", default="predictions.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    return parser


def attach_doi(df, doi_csv):
    if not os.path.exists(doi_csv):
        raise SystemExit(f"--group-by doi needs {doi_csv!r} (not found)")
    ref = pd.read_csv(doi_csv, encoding="utf-8-sig")
    delta_col = "ΔPCE" if "ΔPCE" in ref.columns else "deltaPCE"
    for column in ("smiles", "Initial PCE", delta_col, "DOI number"):
        if column not in ref.columns:
            raise SystemExit(f"{doi_csv!r} is missing column {column!r}")

    def keys(frame, delta):
        return list(zip(frame["smiles"].map(norm_smiles),
                        frame["Initial PCE"].astype(float).round(2),
                        frame[delta].astype(float).round(2)))

    ref = ref.assign(_key=keys(ref, delta_col)).drop_duplicates("_key")
    mapping = dict(zip(ref["_key"], ref["DOI number"].astype(str).str.strip()))
    out = df.copy()
    out["_key"] = keys(out, "deltaPCE")
    out["DOI"] = out["_key"].map(mapping)
    matched = int(out["DOI"].notna().sum())
    print(f"[doi] matched {matched}/{len(out)} rows to a DOI")
    return out.drop(columns=["_key"])


def assign_groups(df, group_by):
    df = df.copy()
    df["group_smiles"] = df["smiles"].map(norm_smiles)
    if group_by == "doi" and "DOI" in df.columns:
        df["group"] = df["DOI"].fillna("noDOI::" + df["group_smiles"])
    else:
        df["group"] = df["group_smiles"]
    print(f"[group] {df['group'].nunique()} groups for {len(df)} rows "
          f"({df['group_smiles'].nunique()} unique molecules)")
    return df


def split_holdout(df, y, groups, test_frac, seed):
    n_splits = max(2, int(round(1.0 / test_frac)))
    for attempt in range(20):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        train_rows, test_rows = next(splitter.split(np.zeros((len(df), 1)), y, groups))
        if len(np.unique(y[test_rows])) == 2:
            break
    else:
        raise SystemExit("Could not build a holdout with both classes; adjust --test-frac/--seed")
    assert_disjoint(groups, train_rows, test_rows, "holdout")
    return train_rows, test_rows


def run_predict(args):
    if not args.run:
        raise SystemExit("--predict needs --run <run_dir>")
    prep = LeakFreePreprocessor.from_dict(joblib.load(os.path.join(args.run, "preprocess.joblib")))
    model = joblib.load(os.path.join(args.run, "model.joblib"))
    df = pd.read_csv(args.predict)
    X_df, data = featurize(df, args.dft_cols)
    prob = model.predict_proba(prep.transform(X_df))[:, 1]
    out = data.copy()
    out["passivator_probability"] = prob
    out["prediction"] = np.where(prob >= 0.5, "PASSIVATOR", "NON-PASSIVATOR")
    out = out.sort_values("passivator_probability", ascending=False)
    out.to_csv(args.predict_out, index=False)
    print(f"[predict] {len(out)} molecules -> {args.predict_out}")
    print(out[["smiles", "passivator_probability", "prediction"]].head(10).to_string(index=False))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.predict:
        return run_predict(args)
    if args.quick:
        args.n_iter = min(args.n_iter, 3)
        args.outer_splits = min(args.outer_splits, 3)

    global N_JOBS, INNER_SPLITS
    N_JOBS = args.n_jobs
    INNER_SPLITS = args.inner_splits
    started = time.time()
    seed = args.seed
    np.random.seed(seed)
    run_dir = args.out or os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"[run] started {time.strftime('%Y-%m-%d %H:%M:%S')} -> {run_dir}")

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise SystemExit(f"{args.data!r} has no column {args.label!r}")
    if args.group_by == "doi":
        df = attach_doi(df, args.doi_csv)
    df = assign_groups(df, args.group_by)

    weights_threshold = args.weight_threshold
    if weights_threshold is None:
        weights_threshold = 2.0 if args.label == "class2.0" else 1.8

    X_df, data = featurize(df, args.dft_cols)
    y = data[args.label].to_numpy().astype(int)
    groups = data["group"].to_numpy()
    delta = data["deltaPCE"].to_numpy(dtype=float)
    weights = None if args.no_weights else compute_sample_weights(delta, weights_threshold)
    log(f"[data] {len(data)} usable rows, {X_df.shape[1]} raw features, "
        f"positive={int((y == 1).sum())}")

    train_rows, test_rows = split_holdout(data, y, groups, args.test_frac, seed)
    log(f"[split] train={len(train_rows)} holdout={len(test_rows)} "
        f"| holdout positive={int((y[test_rows] == 1).sum())}")

    X_train_df = X_df.iloc[train_rows].reset_index(drop=True)
    y_train = y[train_rows]
    g_train = groups[train_rows]
    w_train = None if weights is None else weights[train_rows]

    cs = tuple(float(v) for v in args.lasso_cs.split(",") if v.strip())
    cache = {}

    log("\n[1/4] outer group CV on the training part")
    group_results, group_sizes = run_protocol("group", X_train_df, y_train, g_train, w_train,
                                              args.families, seed, args.outer_splits,
                                              args.n_iter, cache, cs)
    group_table = summarize(group_results)
    log(group_table.to_string(float_format=lambda v: f"{v:.4f}"))

    row_table = None
    if args.leak_check:
        log("\n[2/4] same pipeline with row-wise CV (old protocol, for comparison)")
        row_results, _ = run_protocol("row", X_train_df, y_train, g_train, w_train,
                                      args.families, seed, args.outer_splits,
                                      args.n_iter, cache, cs)
        row_table = summarize(row_results)
        log(row_table.to_string(float_format=lambda v: f"{v:.4f}"))
        merged = group_table.merge(row_table, on="model", suffixes=("_group", "_row"))
        merged["inflation"] = merged["auc_mean_row"] - merged["auc_mean_group"]
        merged = merged.sort_values("inflation", ascending=False)
        log("\n[leak-check] AUC gap (row-wise minus group-wise):")
        log(merged[["model", "auc_mean_group", "auc_mean_row", "inflation"]]
            .to_string(float_format=lambda v: f"{v:.4f}"))
        merged.to_csv(os.path.join(run_dir, "leak_check.csv"), index=False)
    else:
        log("\n[2/4] leak-check skipped (--no-leak-check)")

    log("\n[3/4] selecting the final model by outer group CV AUC")
    base_table = group_table[group_table["model"].isin(args.families)]
    final_name = str(base_table.iloc[0]["model"])
    final_rows = np.arange(len(y_train))
    best_params, _ = tune_family(final_name, X_train_df, y_train, g_train, final_rows,
                                 seed, args.n_iter, cache, w_train, "group", cs)
    preprocessor = get_preprocessor(cache, X_train_df, y_train, final_rows, seed, cs)
    X_fit = preprocessor.transform(X_train_df)
    model = fit_estimator(build_model(final_name, best_params, seed), X_fit, y_train, w_train)
    log(f"[model] {final_name} params={json.dumps(best_params)} "
        f"selected_features={X_fit.shape[1]}")

    log("\n[4/4] holdout evaluation (single shot)")
    X_test = preprocessor.transform(X_df.iloc[test_rows])
    y_test = y[test_rows]
    test_prob = model.predict_proba(X_test)[:, 1]
    holdout = summarize_holdout(y_test, test_prob, seed)
    log(f"[holdout] AUC={holdout['auc']:.4f} "
        f"(95% CI {holdout['auc_ci_low']:.4f}-{holdout['auc_ci_high']:.4f}) "
        f"acc={holdout['accuracy']:.4f} f1={holdout['f1']:.4f} n={holdout['n_samples']}")

    joblib.dump(model, os.path.join(run_dir, "model.joblib"))
    joblib.dump(preprocessor.as_dict(), os.path.join(run_dir, "preprocess.joblib"))
    pd.DataFrame({"smiles": data.iloc[test_rows]["smiles"].to_numpy(),
                  "group": groups[test_rows],
                  "y_true": y_test,
                  "y_prob": test_prob}).to_csv(os.path.join(run_dir, "holdout_predictions.csv"),
                                               index=False)
    with open(os.path.join(run_dir, "selected_features.json"), "w", encoding="utf-8") as handle:
        json.dump(preprocessor.selected_columns, handle, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "config": vars(args),
            "final_model": final_name,
            "final_params": best_params,
            "n_features_raw": int(X_df.shape[1]),
            "n_features_selected": int(X_fit.shape[1]),
            "group_cv": group_table.to_dict(orient="records"),
            "row_cv": None if row_table is None else row_table.to_dict(orient="records"),
            "holdout": holdout,
            "group_cv_selected_sizes": group_sizes,
            "runtime_seconds": round(time.time() - started, 1),
            "data": {"rows": int(len(data)), "train": int(len(train_rows)),
                     "holdout": int(len(test_rows)),
                     "groups": int(len(np.unique(groups)))},
        }, handle, indent=2)
    group_table.to_csv(os.path.join(run_dir, "cv_group.csv"), index=False)
    if row_table is not None:
        row_table.to_csv(os.path.join(run_dir, "cv_row.csv"), index=False)

    log(f"\n[done] artifacts in {run_dir} ({time.time() - started:.0f}s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
