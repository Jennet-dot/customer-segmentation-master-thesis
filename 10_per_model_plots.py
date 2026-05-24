"""
10_per_model_plots.py
---------------------
Generuje szczegółowe wykresy analityczne dla każdego z dziesięciu modeli osobno.

Dla każdego modelu produkowane są cztery typy wykresów:
  (1) Profil RFM klastrów — heatmapa średnich wartości Recency, Frequency,
      Monetary, AvgOrderValue i AvgRating per klaster.
  (2) Rozkład wielkości klastrów — wykres słupkowy z procentami.
  (3) Projekcja PCA 2D — punkty kolorowane wg klastrów. Dla modeli pełnych
      i bazowych projekcja w 32-wymiarowej przestrzeni; dla _noRFM —
      w 26-wymiarowej.
  (4) Profile demograficzne — udziały kluczowych kategorii (kraj,
      kategoria produktu, dochód) w obrębie każdego klastra.

Wszystkie wykresy zapisywane są do podkatalogu results/per_model/ z prefiksem
nazwy modelu (np. fig_RFM_rfm_profile.png).

Uruchomienie:
    python 10_per_model_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import config

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

PER_MODEL_DIR = os.path.join(config.RESULTS_DIR, "per_model")
os.makedirs(PER_MODEL_DIR, exist_ok=True)


# =====================================================================
# Pomocnicze
# =====================================================================
def safe_filename(s: str) -> str:
    """Zamienia spacje i znaki specjalne w nazwie pliku."""
    return s.replace(" ", "_").replace("/", "_")


def labels_to_str(labels) -> pd.Series:
    """Etykiety w jednolitym formacie tekstowym do operacji groupby."""
    return labels.astype(str)


# =====================================================================
# (1) Profil RFM klastrów — heatmapa
# =====================================================================
def plot_rfm_profile(customers: pd.DataFrame, labels: pd.Series, model: str):
    df = customers.loc[labels.index].copy()
    df["__cluster"] = labels_to_str(labels)

    cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "AvgRating", "Age"]
    cols = [c for c in cols if c in df.columns]
    cluster_means = df.groupby("__cluster")[cols].mean()

    # Sortujemy klastry malejąco wg Monetary dla czytelności
    if "Monetary" in cluster_means.columns:
        cluster_means = cluster_means.sort_values("Monetary", ascending=False)

    # Standaryzacja kolumnami do heatmapy (z-score per cecha)
    cm_std = (cluster_means - cluster_means.mean()) / cluster_means.std(ddof=0)
    cm_std = cm_std.fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2),
                                     max(4, len(cluster_means) * 0.4 + 1.5)))
    sns.heatmap(cm_std, annot=cluster_means.round(2), fmt="",
                cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "z-score (kolor)\nwartość średnia (liczba)"},
                linewidths=0.5, linecolor="white")
    ax.set_title(f"Profil cech RFM klastrów — model {model}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cecha")
    ax.set_ylabel("Klaster")
    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"fig_{safe_filename(model)}_01_rfm_profile.png")
    plt.savefig(out)
    plt.close()


# =====================================================================
# (2) Rozkład wielkości klastrów
# =====================================================================
def plot_cluster_sizes(labels: pd.Series, model: str):
    counts = labels.value_counts().sort_index()
    counts.index = counts.index.astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pasek
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = ax1.bar(counts.index, counts.values, color=colors,
                    edgecolor="black", linewidth=0.5)
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + total * 0.005,
                  f"{v:,}\n({100 * v / total:.1f}%)",
                  ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Klaster")
    ax1.set_ylabel("Liczba klientów")
    ax1.set_title(f"Rozkład wielkości klastrów — {model}",
                  fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)

    # Kołowy
    explode = [0.02] * len(counts)
    ax2.pie(counts.values, labels=counts.index, colors=colors,
             autopct="%1.1f%%", startangle=90, explode=explode,
             wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title(f"Procentowy udział klastrów — {model}",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"fig_{safe_filename(model)}_02_sizes.png")
    plt.savefig(out)
    plt.close()


# =====================================================================
# (3) Projekcja PCA 2D
# =====================================================================
def plot_pca_2d(X_df: pd.DataFrame, labels: pd.Series, model: str,
                sample_size: int = 8000):
    rng = np.random.default_rng(config.RANDOM_STATE)
    common = X_df.index.intersection(labels.index)
    if len(common) > sample_size:
        sample_idx = rng.choice(len(common), size=sample_size, replace=False)
        idx = common[sample_idx]
    else:
        idx = common

    X = X_df.loc[idx].values
    lab = labels.loc[idx]

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X2 = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_clusters = sorted(lab.unique(), key=lambda x: str(x))
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique_clusters)))

    for i, c in enumerate(unique_clusters):
        mask = (lab.values == c)
        marker = "x" if str(c) == "-1" else "o"
        alpha = 0.3 if str(c) == "-1" else 0.55
        size = 8 if str(c) == "-1" else 18
        ax.scatter(X2[mask, 0], X2[mask, 1],
                    c=[palette[i % len(palette)]],
                    label=f"Klaster {c} (n={int(mask.sum())})",
                    s=size, alpha=alpha, marker=marker,
                    edgecolors="none")

    ax.set_xlabel(f"PC1 ({100 * var_ratio[0]:.1f}% wariancji)")
    ax.set_ylabel(f"PC2 ({100 * var_ratio[1]:.1f}% wariancji)")
    ax.set_title(f"Projekcja PCA 2D z kolorowaniem klastrów — {model}\n"
                  f"(losowa próbka {len(idx):,} klientów)",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.85)
    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"fig_{safe_filename(model)}_03_pca.png")
    plt.savefig(out)
    plt.close()


# =====================================================================
# (4) Profile demograficzne — udziały kategorii per klaster
# =====================================================================
def plot_demographics(customers: pd.DataFrame, labels: pd.Series, model: str):
    df = customers.loc[labels.index].copy()
    df["__cluster"] = labels_to_str(labels)

    # Wybieramy 3 kluczowe wymiary do pokazania na 3 subplotach
    cat_cols = [("Dom_Country",          "Kraj"),
                ("Dom_Product_Category", "Kategoria produktu"),
                ("Dom_Income",           "Poziom dochodu")]
    cat_cols = [(c, t) for c, t in cat_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5.5))
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, cat_cols):
        # Tabela krzyżowa: klaster x kategoria, znormalizowana po wierszach
        ct = pd.crosstab(df["__cluster"], df[col], normalize="index") * 100
        # Sortujemy klastry wg ich pierwszej dominującej kategorii dla porządku
        ct = ct.sort_index()

        # Stacked bar
        ct.plot(kind="bar", stacked=True, ax=ax,
                colormap="tab20", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Udział (%)")
        ax.set_ylim(0, 100)
        ax.legend(title=title, bbox_to_anchor=(1.02, 1),
                   loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Profile demograficzne klastrów — {model}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"fig_{safe_filename(model)}_04_demographics.png")
    plt.savefig(out)
    plt.close()


# =====================================================================
# (5) Boxploty cech transakcyjnych per klaster
# =====================================================================
def plot_transactional_boxplots(customers: pd.DataFrame, labels: pd.Series,
                                 model: str):
    df = customers.loc[labels.index].copy()
    df["__cluster"] = labels_to_str(labels)

    metrics = [("Recency",       "Recency [dni]"),
               ("Frequency",     "Frequency [transakcje]"),
               ("Monetary",      "Monetary [USD]"),
               ("AvgOrderValue", "AOV [USD]")]
    metrics = [(c, t) for c, t in metrics if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        sns.boxplot(data=df, x="__cluster", y=col, ax=ax,
                     hue="__cluster", legend=False,
                     showfliers=False, palette="Set2")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Klaster")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle(f"Rozkłady cech transakcyjnych w klastrach — {model}",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"fig_{safe_filename(model)}_05_boxplots.png")
    plt.savefig(out)
    plt.close()


# =====================================================================
# Główna procedura
# =====================================================================
def main():
    print("=" * 70)
    print(" 10 — Wykresy analityczne per model")
    print("=" * 70)

    print(f"\n[1/3] Wczytywanie danych ...")
    customers       = pd.read_pickle(config.PATH_CUSTOMERS)
    X_scaled        = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X_scaled_norfm  = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)

    all_labels = {}
    for name in config.MODELS_ORDER:
        path = config.PATH_LABELS[name]
        if os.path.exists(path):
            all_labels[name] = pd.read_pickle(path)

    print(f"      Modeli do przetworzenia: {len(all_labels)}")
    print(f"      Wykresy: 5 typów × {len(all_labels)} modeli = "
          f"{5 * len(all_labels)} plików PNG")

    print(f"\n[2/3] Generowanie wykresów per model ...")
    for i, (model, labels) in enumerate(all_labels.items(), 1):
        # Wybór właściwej przestrzeni cech do PCA
        if model.endswith("_noRFM"):
            X_for_pca = X_scaled_norfm
        else:
            X_for_pca = X_scaled

        print(f"      [{i}/{len(all_labels)}] {model} "
              f"({labels.nunique()} klastrów) ...")
        plot_rfm_profile(customers, labels, model)
        plot_cluster_sizes(labels, model)
        plot_pca_2d(X_for_pca, labels, model)
        plot_demographics(customers, labels, model)
        plot_transactional_boxplots(customers, labels, model)

    print(f"\n[3/3] Gotowe.")
    print(f"\nWszystkie wykresy w: {PER_MODEL_DIR}")
    n_files = sum(1 for f in os.listdir(PER_MODEL_DIR) if f.endswith(".png"))
    print(f"Wygenerowanych plików PNG: {n_files}")


if __name__ == "__main__":
    main()
