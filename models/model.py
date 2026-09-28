"""
Your Doctor - Medical AI Project
=================================
Stage 1: EDA + Data Quality Analysis + Visualization + Initial Preprocessing

Five independent medical modules:
  1. CBC
  2. Diabetes
  3. Liver
  4. Kidney
  5. Thyroid

No ML model is trained in this stage.
Original datasets are never modified.
"""

import warnings
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#4a4a6a",
    "axes.labelcolor": "#e0e0ff",
    "xtick.color": "#a0a0c0",
    "ytick.color": "#a0a0c0",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a",
    "grid.linewidth": 0.5,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#4a4a6a",
    "legend.labelcolor": "#e0e0ff",
    "font.family": "DejaVu Sans",
})

MODULE_PALETTES = {
    "CBC":      sns.color_palette("Set2",  15),
    "Diabetes": sns.color_palette("Set1",  5),
    "Liver":    sns.color_palette("tab10", 10),
    "Kidney":   sns.color_palette("husl",  10),
    "Thyroid":  sns.color_palette("Paired", 10),
}

SEPARATOR = "=" * 75
SUBSEP = "-" * 50


# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_sub(title: str) -> None:
    print(f"\n{SUBSEP}")
    print(f"  {title}")
    print(SUBSEP)


def print_df(df, max_rows: int = 30) -> None:
    with pd.option_context("display.max_columns", None,
                           "display.width", 120,
                           "display.float_format", "{:.4f}".format):
        print(df.to_string(index=True, max_rows=max_rows))


# ---------------------------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------------------------
def save_fig(fig, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHARED ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------
def analyze_dataset(df, module: str) -> None:
    print_sub(f"{module} - Dataset Structure")
    print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\n  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<35} {str(dtype):<12}  unique={df[col].nunique()}")
    print("\n  First 3 rows:")
    print_df(df.head(3))


def analyze_missing_values(df, module: str):
    print_sub(f"{module} - Missing Value Analysis")
    mv = pd.DataFrame({
        "dtype":       df.dtypes,
        "non_null":    df.notnull().sum(),
        "missing":     df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique":      df.nunique(),
    })
    mv = mv[mv["missing"] > 0].sort_values("missing_pct", ascending=False)
    if mv.empty:
        print("  No missing values detected.")
    else:
        print_df(mv)
    return mv


def analyze_duplicates(df, module: str) -> int:
    print_sub(f"{module} - Duplicate Analysis")
    n_dup = df.duplicated().sum()
    print(f"  Total rows    : {len(df)}")
    print(f"  Duplicate rows: {n_dup}  ({n_dup / len(df) * 100:.2f}%)")
    return n_dup


def analyze_target(df, target_col: str, module: str) -> None:
    print_sub(f"{module} - Target Analysis  ['{target_col}']")
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    tbl = pd.DataFrame({"count": counts, "pct_%": pcts.round(2)})
    print_df(tbl)
    max_pct = pcts.max()
    min_pct = pcts.min()
    if max_pct > 70:
        print(f"\n  Class imbalance detected - majority class is {max_pct:.1f}%")
    else:
        print(f"\n  Class balance OK - range {min_pct:.1f}% - {max_pct:.1f}%")


def analyze_numerical_features(df, num_cols: list, module: str):
    print_sub(f"{module} - Numerical Feature Statistics")
    if not num_cols:
        print("  No numerical features.")
        return pd.DataFrame()
    desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skewness"] = df[num_cols].skew().round(3)
    desc["kurtosis"] = df[num_cols].kurtosis().round(3)
    print_df(desc.round(4))
    return desc


def analyze_categorical_features(df, cat_cols: list, module: str) -> None:
    print_sub(f"{module} - Categorical Feature Statistics")
    if not cat_cols:
        print("  No categorical features.")
        return
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        print(f"\n  [{col}]")
        for val, cnt in vc.items():
            print(f"    {str(val):<30} {cnt:>5}  ({cnt / len(df) * 100:.1f}%)")


def analyze_correlations(df, num_cols: list, module: str, threshold: float = 0.60):
    print_sub(f"{module} - Correlation Analysis (Pearson, |r| >= {threshold})")
    if len(num_cols) < 2:
        print("  Not enough numerical features for correlation analysis.")
        return pd.DataFrame()
    corr = df[num_cols].corr(method="pearson")
    strong = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                strong.append({
                    "feature_A": corr.columns[i],
                    "feature_B": corr.columns[j],
                    "pearson_r": round(r, 4),
                })
    if strong:
        strong_df = pd.DataFrame(strong).sort_values(
            "pearson_r", key=abs, ascending=False)
        print_df(strong_df)
        print("\n  High correlations may indicate redundant features."
              " No automatic removal is performed at this stage.")
    else:
        print(f"  No feature pairs with |r| >= {threshold}.")
    return corr


def analyze_outliers(df, num_cols: list, module: str):
    print_sub(f"{module} - Outlier Analysis (IQR method)")
    records = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = ((s < lo) | (s > hi)).sum()
        records.append({
            "feature":     col,
            "Q1":          round(q1, 4),
            "Q3":          round(q3, 4),
            "IQR":         round(iqr, 4),
            "lower_fence": round(lo, 4),
            "upper_fence": round(hi, 4),
            "n_outliers":  n_out,
            "outlier_pct": round(n_out / len(s) * 100, 2),
        })
    out_df = pd.DataFrame(records).sort_values("outlier_pct", ascending=False)
    print_df(out_df)
    print("\n  Medical datasets may contain legitimate extreme values."
          " No automatic removal is performed here.")
    return out_df


# ---------------------------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def _ax_style(ax) -> None:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#a0a0c0", labelsize=8)
    ax.xaxis.label.set_color("#e0e0ff")
    ax.yaxis.label.set_color("#e0e0ff")
    ax.title.set_color("#e0e0ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4a6a")


def plot_target_distribution(df, target_col: str, module: str,
                              out_dir: Path, palette=None) -> None:
    vc = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    fig.suptitle(f"{module} - Target Distribution: '{target_col}'",
                 fontsize=14, color="#e0e0ff", fontweight="bold")
    colors = list(palette[:len(vc)]) if palette else sns.color_palette("Set2", len(vc))
    axes[0].bar(vc.index.astype(str), vc.values, color=colors, edgecolor="#1a1a2e")
    axes[0].set_title("Class Counts")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Count")
    _ax_style(axes[0])
    axes[1].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
                colors=colors, startangle=90,
                textprops={"color": "#e0e0ff", "fontsize": 9})
    axes[1].set_title("Class Proportions")
    _ax_style(axes[1])
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_target_distribution.png")
    print(f"  [OK] Saved: {module.lower()}_target_distribution.png")


def plot_missing_values(df, module: str, out_dir: Path) -> None:
    mv = df.isnull().mean() * 100
    mv = mv[mv > 0].sort_values(ascending=False)
    if mv.empty:
        print(f"  No missing values to visualize for {module}.")
        return
    fig, ax = plt.subplots(figsize=(max(8, len(mv) * 0.6 + 2), 5),
                           facecolor="#1a1a2e")
    colors = [("#e74c3c" if v > 30 else "#f39c12" if v > 10 else "#3498db")
              for v in mv.values]
    ax.barh(mv.index.tolist(), mv.values, color=colors, edgecolor="#1a1a2e")
    ax.set_xlabel("Missing %")
    ax.set_title(f"{module} - Missing Values per Feature")
    ax.axvline(5,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.7, label="5%")
    ax.axvline(30, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="30%")
    ax.legend()
    _ax_style(ax)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_missing_values.png")
    print(f"  [OK] Saved: {module.lower()}_missing_values.png")


def plot_numerical_distributions(df, num_cols: list, module: str,
                                  out_dir: Path, palette=None) -> None:
    if not num_cols:
        return
    cols_to_plot = num_cols[:16]
    n = len(cols_to_plot)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             facecolor="#1a1a2e")
    axes = np.array(axes).flatten()
    pal = list(palette) if palette else sns.color_palette("Set2", n)
    for i, col in enumerate(cols_to_plot):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, color=pal[i % len(pal)],
                     edgecolor="#1a1a2e", alpha=0.85)
        axes[i].set_title(col, fontsize=9)
        _ax_style(axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{module} - Numerical Feature Distributions",
                 fontsize=13, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_num_distributions.png")
    print(f"  [OK] Saved: {module.lower()}_num_distributions.png")


def plot_boxplots(df, num_cols: list, target_col: str, module: str,
                  out_dir: Path, palette=None) -> None:
    if not num_cols:
        return
    cols_to_plot = num_cols[:12]
    n = len(cols_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             facecolor="#1a1a2e")
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols_to_plot):
        if target_col and target_col in df.columns:
            classes = df[target_col].unique()
            groups = [df.loc[df[target_col] == cls, col].dropna().values
                      for cls in classes]
            tick_labels = [str(c) for c in classes]
            try:
                bp = axes[i].boxplot(groups, tick_labels=tick_labels,
                                      patch_artist=True,
                                      medianprops=dict(color="#f39c12", linewidth=2))
            except TypeError:
                bp = axes[i].boxplot(groups, labels=tick_labels,
                                      patch_artist=True,
                                      medianprops=dict(color="#f39c12", linewidth=2))
            pal = list(palette) if palette else sns.color_palette("Set2", len(groups))
            for patch, color in zip(bp["boxes"], pal):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
        else:
            axes[i].boxplot(df[col].dropna().values, patch_artist=True,
                            medianprops=dict(color="#f39c12", linewidth=2))
        axes[i].set_title(col, fontsize=9)
        _ax_style(axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{module} - Boxplots by Target Class",
                 fontsize=13, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_boxplots.png")
    print(f"  [OK] Saved: {module.lower()}_boxplots.png")


def plot_correlation_heatmap(corr, module: str, out_dir: Path) -> None:
    if corr is None or (hasattr(corr, "empty") and corr.empty):
        return
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)),
                           facecolor="#1a1a2e")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=(n <= 15), fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, center=0,
                linewidths=0.3, linecolor="#1a1a2e",
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{module} - Pearson Correlation Heatmap",
                 fontsize=13, color="#e0e0ff", fontweight="bold")
    _ax_style(ax)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_correlation_heatmap.png")
    print(f"  [OK] Saved: {module.lower()}_correlation_heatmap.png")


def plot_feature_vs_target(df, num_cols: list, target_col: str, module: str,
                            out_dir: Path, palette=None, max_features: int = 8) -> None:
    if not num_cols or target_col not in df.columns:
        return
    cols = num_cols[:max_features]
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4),
                             facecolor="#1a1a2e")
    axes = np.array(axes).flatten()
    classes = df[target_col].unique()
    pal = list(palette) if palette else sns.color_palette("Set2", len(classes))
    for i, col in enumerate(cols):
        for j, cls in enumerate(classes):
            data = df.loc[df[target_col] == cls, col].dropna()
            axes[i].hist(data, bins=25, alpha=0.6,
                         color=pal[j % len(pal)], label=str(cls),
                         edgecolor="#1a1a2e")
        axes[i].set_title(f"{col} by {target_col}", fontsize=9)
        axes[i].legend(fontsize=7)
        _ax_style(axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{module} - Feature Distribution by Target Class",
                 fontsize=13, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_feature_vs_target.png")
    print(f"  [OK] Saved: {module.lower()}_feature_vs_target.png")


def plot_scatter_pairs(df, num_cols: list, target_col: str, module: str,
                       out_dir: Path, palette=None, pairs: list = None) -> None:
    if not num_cols or len(num_cols) < 2:
        return
    if pairs is None:
        pairs = [(num_cols[i], num_cols[i + 1])
                 for i in range(0, min(6, len(num_cols) - 1))]
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4.5),
                             facecolor="#1a1a2e")
    axes = np.array(axes).flatten()
    classes = df[target_col].unique() if target_col in df.columns else [None]
    pal = list(palette) if palette else sns.color_palette("Set2", len(classes))
    for i, (x_col, y_col) in enumerate(pairs):
        if x_col not in df.columns or y_col not in df.columns:
            axes[i].set_visible(False)
            continue
        for j, cls in enumerate(classes):
            subset = df[df[target_col] == cls] if cls is not None else df
            subset_xy = subset[[x_col, y_col]].dropna()
            axes[i].scatter(subset_xy[x_col], subset_xy[y_col],
                            alpha=0.5, s=15, color=pal[j % len(pal)],
                            label=str(cls))
        axes[i].set_xlabel(x_col, fontsize=8)
        axes[i].set_ylabel(y_col, fontsize=8)
        axes[i].set_title(f"{x_col} vs {y_col}", fontsize=9)
        if target_col in df.columns:
            axes[i].legend(fontsize=7)
        _ax_style(axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{module} - Scatter Plots",
                 fontsize=13, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_scatter_plots.png")
    print(f"  [OK] Saved: {module.lower()}_scatter_plots.png")


def save_processed_dataset(df, out_path: Path, module: str, n_before: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print_sub(f"{module} - Saved Cleaned Dataset")
    print(f"  Path        : {out_path}")
    print(f"  Rows before : {n_before}")
    print(f"  Rows after  : {len(df)}")
    print(f"  Columns     : {df.shape[1]}")


# ===========================================================================
# MODULE 1 - CBC
# ===========================================================================
def load_cbc_data(root: Path):
    path = root / "datasets" / "CBC" / "diagnosed_cbc_data_v4.csv"
    return pd.read_csv(path)


def preprocess_cbc(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = "Diagnosis"
    df[target_col] = df[target_col].str.strip()
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    dups_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    dups_removed = dups_before - df.duplicated().sum()
    X = df[feature_cols]
    y = df[target_col]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return df, X, y, preprocessor, num_cols, cat_cols, dups_removed


def run_cbc_analysis(root: Path) -> dict:
    MODULE = "CBC"
    print_header(f"MODULE 1 - {MODULE}")
    pal = MODULE_PALETTES[MODULE]
    out_img = root / "datasets" / "CBC" / "processed" / "plots"
    out_csv = root / "datasets" / "CBC" / "processed" / "cbc_cleaned.csv"

    df_raw = load_cbc_data(root)
    n_before = len(df_raw)
    print(f"\n  File : datasets/CBC/diagnosed_cbc_data_v4.csv")
    analyze_dataset(df_raw, MODULE)

    target_col = "Diagnosis"
    analyze_target(df_raw, target_col, MODULE)

    feature_cols = [c for c in df_raw.columns if c != target_col]
    num_cols = df_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_raw[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\n  Numerical features   ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    analyze_missing_values(df_raw, MODULE)
    analyze_duplicates(df_raw, MODULE)
    analyze_numerical_features(df_raw, num_cols, MODULE)
    analyze_categorical_features(df_raw, cat_cols, MODULE)
    analyze_outliers(df_raw, num_cols, MODULE)
    corr = analyze_correlations(df_raw, num_cols, MODULE)

    print_sub(f"{MODULE} - Generating Visualizations")
    plot_target_distribution(df_raw, target_col, MODULE, out_img, pal)
    plot_missing_values(df_raw, MODULE, out_img)
    plot_numerical_distributions(df_raw, num_cols, MODULE, out_img, pal)
    plot_boxplots(df_raw, num_cols, target_col, MODULE, out_img, pal)
    plot_correlation_heatmap(corr, MODULE, out_img)
    plot_feature_vs_target(df_raw, num_cols, target_col, MODULE, out_img, pal)
    cbc_pairs = [("HGB", "HCT"), ("WBC", "NEUTn"), ("RBC", "MCV"),
                 ("MCHC", "MCH"), ("PLT", "PCT")]
    plot_scatter_pairs(df_raw, num_cols, target_col, MODULE, out_img, pal,
                       pairs=[p for p in cbc_pairs
                              if p[0] in num_cols and p[1] in num_cols])

    df_clean, X, y, pre, num_p, cat_p, dups = preprocess_cbc(df_raw)
    save_processed_dataset(df_clean, out_csv, MODULE, n_before)

    return {
        "module":        MODULE,
        "file":          "datasets/CBC/diagnosed_cbc_data_v4.csv",
        "rows_raw":      n_before,
        "cols_raw":      df_raw.shape[1],
        "target":        target_col,
        "n_classes":     df_raw[target_col].nunique(),
        "class_dist":    df_raw[target_col].value_counts().to_dict(),
        "missing_cols":  int(df_raw.isnull().any(axis=0).sum()),
        "duplicates":    int(dups),
        "rows_clean":    len(df_clean),
        "cols_clean":    df_clean.shape[1],
        "num_features":  num_p,
        "cat_features":  cat_p,
        "preprocessing": "RobustScaler + median imputation (numerics)",
    }


# ===========================================================================
# MODULE 2 - DIABETES
# ===========================================================================
def load_diabetes_data(root: Path):
    path = root / "datasets" / "Diabetes" / "Dataset of Diabetes.csv"
    raw = path.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(StringIO(raw))


def preprocess_diabetes(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = "CLASS"
    df[target_col] = df[target_col].astype(str).str.strip().str.upper()
    drop_cols = [c for c in ["ID", "No_Pation"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    dups_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    dups_removed = dups_before - df.duplicated().sum()
    X = df[feature_cols]
    y = df[target_col]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return df, X, y, preprocessor, num_cols, cat_cols, dups_removed


def run_diabetes_analysis(root: Path) -> dict:
    MODULE = "Diabetes"
    print_header(f"MODULE 2 - {MODULE}")
    pal = MODULE_PALETTES[MODULE]
    out_img = root / "datasets" / "Diabetes" / "processed" / "plots"
    out_csv = root / "datasets" / "Diabetes" / "processed" / "diabetes_cleaned.csv"

    df_raw = load_diabetes_data(root)
    n_before = len(df_raw)
    print(f"\n  File : datasets/Diabetes/Dataset of Diabetes.csv")

    if "CLASS" in df_raw.columns:
        df_raw["CLASS"] = df_raw["CLASS"].astype(str).str.strip().str.upper()

    analyze_dataset(df_raw, MODULE)
    target_col = "CLASS"
    analyze_target(df_raw, target_col, MODULE)

    drop_cols = [c for c in ["ID", "No_Pation"] if c in df_raw.columns]
    feature_cols = [c for c in df_raw.columns
                    if c not in [target_col] + drop_cols]
    num_cols = df_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_raw[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\n  Administrative columns dropped from features: {drop_cols}")
    print(f"  Numerical features   ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    analyze_missing_values(df_raw, MODULE)
    analyze_duplicates(df_raw, MODULE)
    analyze_numerical_features(df_raw, num_cols, MODULE)
    analyze_categorical_features(df_raw, cat_cols, MODULE)
    analyze_outliers(df_raw, num_cols, MODULE)
    corr = analyze_correlations(df_raw, num_cols, MODULE)

    print_sub(f"{MODULE} - Generating Visualizations")
    plot_target_distribution(df_raw, target_col, MODULE, out_img, pal)
    plot_missing_values(df_raw, MODULE, out_img)
    plot_numerical_distributions(df_raw, num_cols, MODULE, out_img, pal)
    plot_boxplots(df_raw, num_cols, target_col, MODULE, out_img, pal)
    plot_correlation_heatmap(corr, MODULE, out_img)
    plot_feature_vs_target(df_raw, num_cols, target_col, MODULE, out_img, pal)
    diab_pairs = [("HbA1c", "BMI"), ("Urea", "Cr"), ("Chol", "LDL"),
                  ("TG", "VLDL"), ("HDL", "Chol")]
    plot_scatter_pairs(df_raw, num_cols, target_col, MODULE, out_img, pal,
                       pairs=[p for p in diab_pairs
                              if p[0] in df_raw.columns and p[1] in df_raw.columns])

    df_clean, X, y, pre, num_p, cat_p, dups = preprocess_diabetes(df_raw)
    save_processed_dataset(df_clean, out_csv, MODULE, n_before)

    return {
        "module":        MODULE,
        "file":          "datasets/Diabetes/Dataset of Diabetes.csv",
        "rows_raw":      n_before,
        "cols_raw":      df_raw.shape[1],
        "target":        target_col,
        "n_classes":     df_raw[target_col].nunique(),
        "class_dist":    df_raw[target_col].value_counts().to_dict(),
        "missing_cols":  int(df_raw.isnull().any(axis=0).sum()),
        "duplicates":    int(dups),
        "rows_clean":    len(df_clean),
        "cols_clean":    df_clean.shape[1],
        "num_features":  num_p,
        "cat_features":  cat_p,
        "preprocessing": "RobustScaler + median imputation; ID/No_Pation dropped",
    }


# ===========================================================================
# MODULE 3 - LIVER (ILPD)
# ===========================================================================
def load_liver_data(root: Path):
    path = root / "datasets" / "Liver" / "Indian Liver Patient Dataset (ILPD).csv"
    col_names = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Proteins",
        "Albumin", "Albumin_Globulin_Ratio", "Dataset",
    ]
    return pd.read_csv(path, header=None, names=col_names)


def preprocess_liver(df):
    df = df.copy()
    df["Liver_Patient"] = (df["Dataset"] == 1).astype(int)
    df.drop(columns=["Dataset"], inplace=True)
    target_col = "Liver_Patient"
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    dups_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    dups_removed = dups_before - df.duplicated().sum()
    X = df[feature_cols]
    y = df[target_col]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return df, X, y, preprocessor, num_cols, cat_cols, dups_removed


def run_liver_analysis(root: Path) -> dict:
    MODULE = "Liver"
    print_header(f"MODULE 3 - {MODULE} (ILPD)")
    pal = MODULE_PALETTES[MODULE]
    out_img = root / "datasets" / "Liver" / "processed" / "plots"
    out_csv = root / "datasets" / "Liver" / "processed" / "liver_cleaned.csv"

    df_raw = load_liver_data(root)
    n_before = len(df_raw)
    print(f"\n  File : datasets/Liver/Indian Liver Patient Dataset (ILPD).csv")
    print("  Note : No header in original file - column names from UCI documentation.")
    print("  Target 'Dataset': 1 = liver patient, 2 = non-patient -> mapped to 1/0")

    analyze_dataset(df_raw, MODULE)
    raw_target = "Dataset"
    analyze_target(df_raw, raw_target, MODULE)

    feature_cols = [c for c in df_raw.columns if c != raw_target]
    num_cols = df_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_raw[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\n  Numerical features   ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    analyze_missing_values(df_raw, MODULE)
    analyze_duplicates(df_raw, MODULE)
    analyze_numerical_features(df_raw, num_cols, MODULE)
    analyze_categorical_features(df_raw, cat_cols, MODULE)
    analyze_outliers(df_raw, num_cols, MODULE)
    corr = analyze_correlations(df_raw, num_cols, MODULE)

    print_sub(f"{MODULE} - Generating Visualizations")
    plot_target_distribution(df_raw, raw_target, MODULE, out_img, pal)
    plot_missing_values(df_raw, MODULE, out_img)
    plot_numerical_distributions(df_raw, num_cols, MODULE, out_img, pal)
    plot_boxplots(df_raw, num_cols, raw_target, MODULE, out_img, pal)
    plot_correlation_heatmap(corr, MODULE, out_img)
    plot_feature_vs_target(df_raw, num_cols, raw_target, MODULE, out_img, pal)
    liver_pairs = [
        ("Total_Bilirubin",          "Direct_Bilirubin"),
        ("Alamine_Aminotransferase", "Aspartate_Aminotransferase"),
        ("Total_Proteins",           "Albumin"),
        ("Albumin",                  "Albumin_Globulin_Ratio"),
        ("Alkaline_Phosphotase",     "Alamine_Aminotransferase"),
    ]
    plot_scatter_pairs(df_raw, num_cols, raw_target, MODULE, out_img, pal,
                       pairs=[p for p in liver_pairs
                              if p[0] in df_raw.columns and p[1] in df_raw.columns])

    df_clean, X, y, pre, num_p, cat_p, dups = preprocess_liver(df_raw)
    save_processed_dataset(df_clean, out_csv, MODULE, n_before)

    return {
        "module":        MODULE,
        "file":          "datasets/Liver/Indian Liver Patient Dataset (ILPD).csv",
        "rows_raw":      n_before,
        "cols_raw":      df_raw.shape[1],
        "target":        "Liver_Patient (mapped: Dataset 1->1, 2->0)",
        "n_classes":     2,
        "class_dist":    df_raw[raw_target].value_counts().to_dict(),
        "missing_cols":  int(df_raw.isnull().any(axis=0).sum()),
        "duplicates":    int(dups),
        "rows_clean":    len(df_clean),
        "cols_clean":    df_clean.shape[1],
        "num_features":  num_p,
        "cat_features":  cat_p,
        "preprocessing": "RobustScaler + median imputation; target mapped 1->1, 2->0",
    }


# ===========================================================================
# MODULE 4 - KIDNEY (ARFF)
# ===========================================================================
def _parse_arff(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    attributes = []
    data_section = False
    data_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        low = stripped.lower()
        if low.startswith("@attribute"):
            parts = stripped.split()
            name_raw = parts[1].strip("'\"")
            attributes.append(name_raw)
        elif low.startswith("@data"):
            data_section = True
        elif data_section and stripped:
            tokens = [t.strip() for t in stripped.split(",")]
            if len(tokens) > len(attributes):
                tokens = [t for t in tokens if t]
            if len(tokens) > len(attributes):
                tokens = tokens[:len(attributes)]
            if len(tokens) < len(attributes):
                tokens += [np.nan] * (len(attributes) - len(tokens))
            data_rows.append(tokens)
    df = pd.DataFrame(data_rows, columns=attributes)
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def load_kidney_data(root: Path):
    path = root / "datasets" / "Kidney" / "chronic_kidney_disease.arff"
    return _parse_arff(path), path.name


def _coerce_kidney_types(df):
    numeric_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
                    "hemo", "pcv", "wbcc", "rbcc", "sg", "al", "su"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess_kidney(df):
    df = df.copy()
    target_col = "class"
    num_def = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
               "hemo", "pcv", "wbcc", "rbcc", "sg", "al", "su"]
    cat_def = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    num_cols = [c for c in num_def if c in df.columns]
    cat_cols = [c for c in cat_def if c in df.columns]
    if "dm" in df.columns:
        df["dm"] = df["dm"].str.replace(r"\s+", "", regex=True).str.lower()
        df["dm"] = df["dm"].map({"yes": "yes", "no": "no"})
    if "cad" in df.columns:
        df["cad"] = df["cad"].str.replace(r"\s+", "", regex=True).str.lower()
    X = df[[c for c in num_cols + cat_cols if c in df.columns]]
    y = df[target_col]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_cols),
                                       ("cat", cat_pipe, cat_cols)],
                                      remainder="drop")
    return df, X, y, preprocessor, num_cols, cat_cols


def run_kidney_analysis(root: Path) -> dict:
    MODULE = "Kidney"
    print_header(f"MODULE 4 - {MODULE} (CKD - ARFF)")
    pal = MODULE_PALETTES[MODULE]
    out_img = root / "datasets" / "Kidney" / "processed" / "plots"
    out_csv = root / "datasets" / "Kidney" / "processed" / "kidney_cleaned.csv"

    df_raw, arff_file = load_kidney_data(root)
    n_before = len(df_raw)
    print(f"\n  File used : datasets/Kidney/{arff_file}")
    print("  Rationale : chronic_kidney_disease.arff and _full.arff both contain")
    print("              400 instances with identical 25 attributes. The non-full file")
    print("              is the canonical clean ARFF; _full adds only extra comments.")
    print("  Missing   : '?' markers converted to NaN during ARFF parsing.")

    df_raw = _coerce_kidney_types(df_raw)
    analyze_dataset(df_raw, MODULE)

    target_col = "class"
    analyze_target(df_raw, target_col, MODULE)

    num_def = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
               "hemo", "pcv", "wbcc", "rbcc", "sg", "al", "su"]
    cat_def = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    num_cols = [c for c in num_def if c in df_raw.columns]
    cat_cols = [c for c in cat_def if c in df_raw.columns]
    print(f"\n  Numerical features   ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    analyze_missing_values(df_raw, MODULE)
    analyze_duplicates(df_raw, MODULE)
    analyze_numerical_features(df_raw, num_cols, MODULE)
    analyze_categorical_features(df_raw, cat_cols, MODULE)
    analyze_outliers(df_raw, num_cols, MODULE)
    corr = analyze_correlations(df_raw, num_cols, MODULE)

    print_sub(f"{MODULE} - Generating Visualizations")
    plot_target_distribution(df_raw, target_col, MODULE, out_img, pal)
    plot_missing_values(df_raw, MODULE, out_img)
    plot_numerical_distributions(df_raw, num_cols, MODULE, out_img, pal)
    plot_boxplots(df_raw, num_cols, target_col, MODULE, out_img, pal)
    plot_correlation_heatmap(corr, MODULE, out_img)
    plot_feature_vs_target(df_raw, num_cols, target_col, MODULE, out_img, pal)
    kidney_pairs = [("sc", "bu"), ("hemo", "pcv"), ("sod", "pot"),
                    ("bgr", "sc"), ("age", "bp"), ("wbcc", "rbcc")]
    plot_scatter_pairs(df_raw, num_cols, target_col, MODULE, out_img, pal,
                       pairs=[p for p in kidney_pairs
                              if p[0] in num_cols and p[1] in num_cols])

    df_clean, X, y, pre, num_p, cat_p = preprocess_kidney(df_raw)
    save_processed_dataset(df_clean, out_csv, MODULE, n_before)

    return {
        "module":        MODULE,
        "file":          f"datasets/Kidney/{arff_file}",
        "rows_raw":      n_before,
        "cols_raw":      df_raw.shape[1],
        "target":        target_col,
        "n_classes":     df_raw[target_col].nunique(),
        "class_dist":    df_raw[target_col].value_counts().to_dict(),
        "missing_cols":  int(df_raw.isnull().any(axis=0).sum()),
        "duplicates":    int(df_raw.duplicated().sum()),
        "rows_clean":    len(df_clean),
        "cols_clean":    df_clean.shape[1],
        "num_features":  num_p,
        "cat_features":  cat_p,
        "preprocessing": "RobustScaler + median/mode imputation; ? -> NaN; dm/cad whitespace fix",
    }


# ===========================================================================
# MODULE 5 - THYROID (ANN dataset)
# ===========================================================================
def load_thyroid_data(root: Path):
    """
    Files: ann-train.data (3772 rows) and ann-test.data (3428 rows).

    Rationale:
      ann-thyroid.names and ann-Readme document these as the official split for
      the Daimler-Benz thyroid ANN dataset (3 classes).
      Other files in the directory (allhypo, allhyper, allbp, sick, hypothyroid,
      dis, etc.) are completely different UCI thyroid subsets with different
      targets, feature sets, and formats. They are excluded.
      The official train/test split is preserved; files are NOT randomly merged.

    Feature layout (21 inputs + 1 target = 22 columns, space-delimited):
      Continuous (positions 0,16-20): age_norm, TSH_norm, T3_norm, TT4_norm, T4U_norm, FTI_norm
      Binary (positions 1-15): sex + 14 clinical indicator flags
      Target (position 21): 1=normal, 2=hyperfunction, 3=subnormal functioning
    """
    col_names = [
        "age_norm", "sex", "on_thyroxine", "query_on_thyroxine",
        "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
        "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
        "lithium", "goitre", "tumor", "hypopituitary", "psych",
        "TSH_norm", "T3_norm", "TT4_norm", "T4U_norm", "FTI_norm", "target",
    ]

    def _read_space_file(path):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) == len(col_names):
                rows.append(tokens)
        df = pd.DataFrame(rows, columns=col_names)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    train_path = root / "datasets" / "Thyroid" / "ann-train.data"
    test_path  = root / "datasets" / "Thyroid" / "ann-test.data"
    df_train = _read_space_file(train_path)
    df_test  = _read_space_file(test_path)
    df_train["split"] = "train"
    df_test["split"]  = "test"
    return df_train, df_test


def _map_thyroid_target(df):
    df = df.copy()
    mapping = {1: "normal", 2: "hyperfunction", 3: "subnormal_functioning"}
    df["target_label"] = df["target"].map(mapping)
    return df


def preprocess_thyroid(df_train, df_test, num_cols: list, binary_cols: list):
    target_col = "target"
    feature_cols = num_cols + binary_cols
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test  = df_test[feature_cols]
    y_test  = df_test[target_col]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_cols),
                                       ("bin", bin_pipe, binary_cols)],
                                      remainder="drop")
    return preprocessor, X_train, y_train, X_test, y_test


def _plot_thyroid_pairplot(df, num_cols, target_col, module, out_dir, palette) -> None:
    if len(num_cols) < 2:
        return
    sub = df[num_cols + [target_col]].dropna()
    if sub.empty:
        return
    classes = sub[target_col].unique()
    pal_dict = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}
    n = len(num_cols)
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3), facecolor="#1a1a2e")
    for i, col_y in enumerate(num_cols):
        for j, col_x in enumerate(num_cols):
            ax = axes[i][j]
            if i == j:
                for cls in classes:
                    d = sub.loc[sub[target_col] == cls, col_x].dropna()
                    ax.hist(d, bins=20, alpha=0.6,
                            color=pal_dict.get(cls, "#888"), density=True)
            else:
                for cls in classes:
                    d = sub[sub[target_col] == cls]
                    ax.scatter(d[col_x], d[col_y], s=5, alpha=0.4,
                               color=pal_dict.get(cls, "#888"))
            if i == n - 1:
                ax.set_xlabel(col_x, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(col_y, fontsize=7)
            else:
                ax.set_yticklabels([])
            _ax_style(ax)
    fig.suptitle(f"{module} - Pairplot of Continuous Features",
                 fontsize=12, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_pairplot.png")
    print(f"  [OK] Saved: {module.lower()}_pairplot.png")


def _plot_thyroid_binary_flags(df, binary_cols, target_col, module, out_dir, palette) -> None:
    if not binary_cols:
        return
    cols_plot = binary_cols[:12]
    classes = df[target_col].unique()
    n = len(cols_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             facecolor="#1a1a2e")
    axes = np.array(axes).flatten()
    pal = list(palette)[:len(classes)]
    for i, col in enumerate(cols_plot):
        rates = [df.loc[df[target_col] == cls, col].mean() for cls in classes]
        axes[i].bar([str(c) for c in classes], rates, color=pal, edgecolor="#1a1a2e")
        axes[i].set_title(f"{col} flag rate", fontsize=8)
        axes[i].set_ylabel("Mean proportion")
        _ax_style(axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{module} - Binary Flag Rates by Target Class",
                 fontsize=12, color="#e0e0ff", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, f"{module.lower()}_binary_flags.png")
    print(f"  [OK] Saved: {module.lower()}_binary_flags.png")


def run_thyroid_analysis(root: Path) -> dict:
    MODULE = "Thyroid"
    print_header(f"MODULE 5 - {MODULE} (ANN Dataset)")
    pal = MODULE_PALETTES[MODULE]
    out_img   = root / "datasets" / "Thyroid" / "processed" / "plots"
    out_train = root / "datasets" / "Thyroid" / "processed" / "thyroid_train_cleaned.csv"
    out_test  = root / "datasets" / "Thyroid" / "processed" / "thyroid_test_cleaned.csv"
    out_full  = root / "datasets" / "Thyroid" / "processed" / "thyroid_cleaned.csv"

    df_train, df_test = load_thyroid_data(root)
    print("\n  Files used:")
    print("    ann-train.data  (3772 training instances)")
    print("    ann-test.data   (3428 testing  instances)")
    print("  Official train/test split preserved - NOT randomly re-split.")
    print("  Other thyroid files (allhypo, allhyper, sick, hypothyroid, etc.)")
    print("  are different UCI subsets with different targets and are excluded.")
    print("  Target: 1=normal, 2=hyperfunction, 3=subnormal functioning")

    df_train = _map_thyroid_target(df_train)
    df_test  = _map_thyroid_target(df_test)
    df_all   = pd.concat([df_train, df_test], ignore_index=True)
    n_before = len(df_all)

    analyze_dataset(df_all, MODULE)

    target_col  = "target_label"
    num_cols = [c for c in ["age_norm", "TSH_norm", "T3_norm",
                             "TT4_norm", "T4U_norm", "FTI_norm"]
                if c in df_all.columns]
    binary_cols = [c for c in [
        "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment",
        "query_hypothyroid", "query_hyperthyroid", "lithium",
        "goitre", "tumor", "hypopituitary", "psych",
    ] if c in df_all.columns]

    print(f"\n  Numerical features  ({len(num_cols)}): {num_cols}")
    print(f"  Binary features     ({len(binary_cols)}): {binary_cols}")
    print(f"  Train rows: {len(df_train)}   Test rows: {len(df_test)}")

    analyze_target(df_all, target_col, MODULE)
    analyze_missing_values(df_all, MODULE)
    analyze_duplicates(df_all, MODULE)
    analyze_numerical_features(df_all, num_cols, MODULE)
    analyze_categorical_features(df_all, binary_cols[:8], MODULE)
    analyze_outliers(df_all, num_cols, MODULE)
    corr = analyze_correlations(df_all, num_cols, MODULE)

    print_sub(f"{MODULE} - Generating Visualizations")
    plot_target_distribution(df_all, target_col, MODULE, out_img, pal)
    plot_missing_values(df_all, MODULE, out_img)
    plot_numerical_distributions(df_all, num_cols, MODULE, out_img, pal)
    plot_boxplots(df_all, num_cols, target_col, MODULE, out_img, pal)
    plot_correlation_heatmap(corr, MODULE, out_img)
    plot_feature_vs_target(df_all, num_cols, target_col, MODULE, out_img, pal)
    thyroid_pairs = [("TSH_norm", "T3_norm"), ("TT4_norm", "T4U_norm"),
                     ("FTI_norm", "TSH_norm"), ("T3_norm", "TT4_norm"),
                     ("age_norm", "TSH_norm")]
    plot_scatter_pairs(df_all, num_cols, target_col, MODULE, out_img, pal,
                       pairs=[p for p in thyroid_pairs
                              if p[0] in num_cols and p[1] in num_cols])
    _plot_thyroid_pairplot(df_all, num_cols, target_col, MODULE, out_img, pal)
    _plot_thyroid_binary_flags(df_all, binary_cols, target_col, MODULE, out_img, pal)

    preprocessor, X_train, y_train, X_test, y_test = preprocess_thyroid(
        df_train, df_test, num_cols, binary_cols)

    df_train_clean = df_train.drop(columns=["split"], errors="ignore")
    df_test_clean  = df_test.drop(columns=["split"],  errors="ignore")
    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train_clean.to_csv(out_train, index=False)
    df_test_clean.to_csv(out_test,   index=False)
    df_all_clean = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
    df_all_clean.to_csv(out_full, index=False)

    print_sub(f"{MODULE} - Saved Cleaned Datasets")
    print(f"  Train : {out_train}  ({len(df_train_clean)} rows)")
    print(f"  Test  : {out_test}  ({len(df_test_clean)} rows)")
    print(f"  Full  : {out_full}  ({len(df_all_clean)} rows)")

    return {
        "module":        MODULE,
        "files":         "ann-train.data + ann-test.data (official split preserved)",
        "rows_train":    len(df_train),
        "rows_test":     len(df_test),
        "rows_total":    len(df_all),
        "cols":          df_all.shape[1],
        "target":        target_col,
        "n_classes":     df_all[target_col].nunique(),
        "class_dist":    df_all[target_col].value_counts().to_dict(),
        "missing_cols":  int(df_all.isnull().any(axis=0).sum()),
        "duplicates":    int(df_all.duplicated().sum()),
        "num_features":  num_cols,
        "bin_features":  binary_cols,
        "preprocessing": "StandardScaler on continuous; mode imputation on binary; train/test split preserved",
    }


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
def print_final_summary(summaries: list) -> None:
    print_header("OVERALL PROJECT SUMMARY - YOUR DOCTOR")
    print("  Five independent medical modules completed EDA + Preprocessing.\n")
    for s in summaries:
        print(SUBSEP)
        print(f"  DATASET : {s['module']}")
        print(f"  File    : {s.get('file', s.get('files', 'N/A'))}")
        rows = s.get("rows_raw", s.get("rows_total", "N/A"))
        print(f"  Rows    : {rows}  |  Cols: {s.get('cols_raw', s.get('cols', 'N/A'))}")
        print(f"  Target  : {s['target']}")
        print(f"  Classes : {s['n_classes']}  ->  {s['class_dist']}")
        print(f"  Missing : {s['missing_cols']} columns with missing values")
        print(f"  Dupes   : {s['duplicates']}")
        if s.get("rows_train"):
            print(f"  Train/Test split : {s['rows_train']} / {s['rows_test']} (preserved)")
        clean_rows = s.get("rows_clean", s.get("rows_total", "N/A"))
        clean_cols = s.get("cols_clean", s.get("cols", "N/A"))
        print(f"  Clean   : {clean_rows} rows x {clean_cols} cols")
        print(f"  Preprocessing: {s['preprocessing']}")

    print(f"\n{SEPARATOR}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(SEPARATOR)
    hdr = (f"{'Module':<12} {'Rows':>6} {'Cols':>5} "
           f"{'Target':<38} {'Classes':>7} {'MissCols':>8} {'Dupes':>6}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        rows = s.get("rows_raw", s.get("rows_total", "?"))
        cols = s.get("cols_raw", s.get("cols", "?"))
        tgt  = str(s["target"])[:38]
        print(f"{s['module']:<12} {rows:>6} {cols:>5} {tgt:<38} "
              f"{s['n_classes']:>7} {s['missing_cols']:>8} {s['duplicates']:>6}")
    print(SEPARATOR)
    print("\n  Stage 1 complete: EDA + Data Quality + Visualization + Preprocessing")
    print("  Cleaned datasets saved to datasets/<Module>/processed/")
    print("  Visualisations saved to datasets/<Module>/processed/plots/")
    print("  Lets start modelling ya Omryyyyyyy.")
    print(SEPARATOR)


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    root = get_project_root()
    print(SEPARATOR)
    print("  YOUR DOCTOR - Stage 1: EDA + Data Quality + Visualization + Preprocessing")
    print(f"  Project root : {root}")
    print(SEPARATOR)
    summaries = []
    summaries.append(run_cbc_analysis(root))
    summaries.append(run_diabetes_analysis(root))
    summaries.append(run_liver_analysis(root))
    summaries.append(run_kidney_analysis(root))
    summaries.append(run_thyroid_analysis(root))
    print_final_summary(summaries)


if __name__ == "__main__":
    main()
