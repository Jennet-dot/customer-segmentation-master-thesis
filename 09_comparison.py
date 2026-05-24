"""
09_comparison.py
----------------
Końcowe porównanie modeli — tabela zbiorcza + wykresy do pracy magisterskiej.

Generuje:
  - summary_table.csv         — zbiorcza tabela wszystkich modeli i mierników
  - fig_kmeans_search.png     — metoda łokcia i silhouette dla K-means
  - fig_gmm_bic.png           — BIC/AIC dla GMM
  - fig_silhouette_compare.png — silhouette wszystkich modeli
  - fig_internal_metrics.png  — porównanie wewnętrznych mierników
  - fig_external_metrics.png  — porównanie zewnętrznych mierników
  - fig_business_srs.png      — SRS per segment per model
  - fig_business_aov.png      — AOV per segment per model
  - fig_business_churn.png    — CR per segment per model
  - fig_pairwise_ari_heatmap.png — heatmapa ARI między modelami

Uruchomienie:
    python 09_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def build_summary_table() -> pd.DataFrame:
    """Łączy mierniki wewnętrzne (własna przestrzeń) i zewnętrzne w jedną tabelę."""
    internal = pd.read_csv(os.path.join(config.RESULTS_DIR, "internal_metrics.csv"),
                            index_col="model")
    external = pd.read_csv(os.path.join(config.RESULTS_DIR, "external_metrics.csv"),
                            index_col="model")
    rar = pd.read_csv(os.path.join(config.RESULTS_DIR, "rar_global.csv"),
                       index_col="model")
    summary = internal.join(external).join(rar)
    rounding = {
        "silhouette": 4, "calinski_harabasz": 2, "davies_bouldin": 4,
        "WSS": 2, "BSS": 2, "dunn": 4,
        "ARI": 4, "purity": 4, "entropy": 4, "VI": 4, "RaR_global_pct": 2,
    }
    for col, dec in rounding.items():
        if col in summary.columns:
            summary[col] = summary[col].round(dec)
    summary = summary.reindex([m for m in config.MODELS_ORDER if m in summary.index])
    return summary


def build_summary_table_pca() -> pd.DataFrame:
    """Tabela zbiorcza z miernikami w przestrzeni PCA (wspólna baza porównawcza)."""
    pca_path = os.path.join(config.RESULTS_DIR, "internal_metrics_pca.csv")
    if not os.path.exists(pca_path):
        return None
    internal = pd.read_csv(pca_path, index_col="model")
    external = pd.read_csv(os.path.join(config.RESULTS_DIR, "external_metrics.csv"),
                            index_col="model")
    rar = pd.read_csv(os.path.join(config.RESULTS_DIR, "rar_global.csv"),
                       index_col="model")
    summary = internal.join(external).join(rar)
    for col in ["silhouette","calinski_harabasz","davies_bouldin","ARI","purity"]:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    return summary.reindex([m for m in config.MODELS_ORDER if m in summary.index])


def plot_kmeans_search():
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "kmeans_search.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["k"], df["inertia"], "o-", color="#2c3e50", linewidth=2)
    axes[0].set_xlabel("Liczba klastrów k")
    axes[0].set_ylabel("Inercja (WCSS)")
    axes[0].set_title("Metoda łokcia — K-means")
    axes[0].set_xticks(df["k"])

    axes[1].plot(df["k"], df["silhouette"], "o-", color="#c0392b", linewidth=2)
    best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
    axes[1].axvline(best_k, color="green", linestyle="--", alpha=0.6,
                     label=f"Najlepsze k={best_k}")
    axes[1].set_xlabel("Liczba klastrów k")
    axes[1].set_ylabel("Współczynnik silhouette")
    axes[1].set_title("Silhouette — K-means")
    axes[1].set_xticks(df["k"])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_kmeans_search.png"))
    plt.close()


def plot_gmm_bic():
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "gmm_search.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["k"], df["bic"], "o-", label="BIC", color="#2980b9", linewidth=2)
    ax.plot(df["k"], df["aic"], "s-", label="AIC", color="#e67e22", linewidth=2)
    best_k = int(df.loc[df["bic"].idxmin(), "k"])
    ax.axvline(best_k, color="green", linestyle="--", alpha=0.6,
                label=f"Najlepsze K={best_k} (min BIC)")
    ax.set_xlabel("Liczba komponentów K")
    ax.set_ylabel("Wartość kryterium")
    ax.set_title("Dobór K dla GMM (BIC vs AIC)")
    ax.set_xticks(df["k"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_gmm_bic.png"))
    plt.close()


def plot_internal_metrics(summary: pd.DataFrame):
    """
    Porównanie 4 dimensionless mierników wewnętrznych na 4 subplotach.

    UWAGA: pominęliśmy WSS i BSS, ponieważ są one wrażliwe na liczbę cech.
    Każdy model jest oceniany w swojej własnej przestrzeni (RFM → 3D, full → 32D,
    noRFM → 26D), więc WSS i BSS są nieporównywalne między różnymi przestrzeniami
    (w 32D są ~10× większe niż w 3D z czystych geometrycznych powodów).
    Pełne wartości WSS i BSS pozostają w pliku internal_metrics.csv dla
    zaawansowanych analiz w obrębie tej samej przestrzeni.

    Dunn pokazujemy na skali logarytmicznej, bo wartości RFM/wRFM (~6e-5)
    są o rząd wielkości niższe od pozostałych modeli (~0.1-0.3) — wynika to
    z geometrii: w 3D z 86k klientami klastry kwartylowe stykają się ze sobą.
    """
    metrics = [("silhouette",        "Silhouette (↑ lepiej)",        "Greens", False),
                ("calinski_harabasz", "Calinski-Harabasz (↑ lepiej)", "Greens", False),
                ("davies_bouldin",    "Davies-Bouldin (↓ lepiej)",    "Reds",   False),
                ("dunn",              "Dunn (↑ lepiej, skala log)",   "Greens", True)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (col, title, cmap, use_log) in zip(axes.ravel(), metrics):
        vals = summary[col]
        colors = sns.color_palette(cmap, n_colors=len(vals))
        ax.bar(vals.index, vals.values, color=colors,
                edgecolor="black", linewidth=0.5)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(col)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    plt.suptitle("Wewnętrzne mierniki jakości klasteryzacji\n"
                 "(każdy model oceniany w swojej własnej przestrzeni cech)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_internal_metrics.png"))
    plt.close()


def plot_external_metrics(summary: pd.DataFrame):
    """Porównanie 4 mierników zewnętrznych."""
    metrics = [("ARI",     "ARI (↑ lepiej)",      "Blues"),
                ("purity",  "Purity (↑ lepiej)",   "Blues"),
                ("entropy", "Entropy (↓ lepiej)",  "Oranges"),
                ("VI",      "VI (↓ lepiej)",       "Oranges")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (col, title, cmap) in zip(axes.ravel(), metrics):
        vals = summary[col]
        colors = sns.color_palette(cmap, n_colors=len(vals))
        ax.bar(vals.index, vals.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(col)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    plt.suptitle("Zewnętrzne mierniki jakości klasteryzacji (vs Customer_Segment)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_external_metrics.png"))
    plt.close()


def plot_business(metric_col: str, title: str, fname: str, ylabel: str):
    """Wykres słupkowy wybranego miernika biznesowego per segment per model."""
    biz = pd.read_csv(os.path.join(config.RESULTS_DIR, "business_metrics.csv"))
    models = [m for m in config.MODELS_ORDER if m in biz["model"].unique()]
    n_models = len(models)

    # Layout dynamiczny — 5 kolumn, tyle wierszy ile potrzeba
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = axes.ravel() if n_models > 1 else [axes]

    for ax, model in zip(axes_flat, models):
        sub = biz[biz["model"] == model].copy()
        # Sortuj segmenty po wartości miernika dla czytelności
        sub = sub.sort_values(metric_col, ascending=False)
        colors = sns.color_palette("viridis", n_colors=len(sub))
        ax.bar(range(len(sub)), sub[metric_col].values, color=colors,
                edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["segment"].astype(str).values,
                            rotation=35, ha="right", fontsize=8)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)

    # Ukryj puste subplots
    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, fname))
    plt.close()


def plot_pairwise_ari():
    pari = pd.read_csv(os.path.join(config.RESULTS_DIR, "pairwise_ari.csv"),
                        index_col=0)
    pari = pari.reindex(index=config.MODELS_ORDER, columns=config.MODELS_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(pari, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-0.1, vmax=1.0, center=0.0, square=True,
                cbar_kws={"label": "Adjusted Rand Index"},
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Pairwise ARI — zgodność modeli między sobą",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_pairwise_ari_heatmap.png"))
    plt.close()


def plot_silhouette_dual(summary_own: pd.DataFrame, summary_pca: pd.DataFrame):
    """
    Dwa panele silhouette: własna przestrzeń (lewy) vs PCA-15 (prawy).
    Pokazuje jak curse of dimensionality zawyżał różnice między modelami.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax, (df, title, subtitle) in zip(axes, [
        (summary_own,
         "Silhouette — własna przestrzeń każdego modelu",
         "RFM/wRFM: 3D  |  zaawansowane full: 32D  |  noRFM: 26D"),
        (summary_pca,
         "Silhouette — wspólna przestrzeń PCA-15 (77% wariancji)",
         "Wszystkie modele oceniane w tej samej zredukowanej przestrzeni"),
    ]):
        vals = df["silhouette"].fillna(0)
        colors = sns.color_palette("Set2", n_colors=len(vals))
        bars = ax.bar(range(len(vals)), vals.values,
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, k, s in zip(bars, df["n_clusters"].values, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), 0) + 0.003,
                    f"k={int(k)}\n{s:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Silhouette")
        ax.set_title(f"{title}\n({subtitle})", fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(min(vals.min() - 0.02, -0.02), vals.max() * 1.25)

    plt.suptitle("Porównanie silhouette w dwóch przestrzeniach ewaluacji",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_silhouette_dual_space.png"))
    plt.close()


def plot_feature_ablation(summary: pd.DataFrame):
    """
    Najważniejszy wykres odpowiadający na pytanie badawcze:
    Dla każdego z 4 zaawansowanych algorytmów porównuje wyniki dla 2 wariantów cech:
      - 'pełne' (32 cechy, włącznie z RFM)
      - 'bez RFM' (26 cech demograficzno-behawioralnych)
    Pokazuje 4 wskaźniki obok siebie: silhouette, ARI vs ground truth, n_clusters,
    Davies-Bouldin (niższe = lepsze).
    """
    pairs = [("KMeans", "KMeans_noRFM"),
             ("Hierarchical", "Hierarchical_noRFM"),
             ("HDBSCAN", "HDBSCAN_noRFM"),
             ("GMM", "GMM_noRFM")]
    pairs = [(a, b) for a, b in pairs if a in summary.index and b in summary.index]
    if not pairs:
        return

    metrics_to_plot = [
        ("silhouette",     "Silhouette (↑ lepiej)"),
        ("ARI",            "ARI vs Customer_Segment (↑ lepiej)"),
        ("davies_bouldin", "Davies-Bouldin (↓ lepiej)"),
        ("n_clusters",     "Liczba klastrów"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(pairs))
    width = 0.36

    for ax, (col, title) in zip(axes, metrics_to_plot):
        full_vals = [summary.loc[a, col] for a, _ in pairs]
        norfm_vals = [summary.loc[b, col] for _, b in pairs]

        ax.bar(x - width / 2, full_vals, width, label="Pełne cechy (32)",
               color="#2980b9", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, norfm_vals, width, label="Bez RFM (26)",
               color="#e67e22", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([a for a, _ in pairs], rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

    plt.suptitle("Wpływ usunięcia cech RFM na jakość segmentacji zaawansowanych modeli",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "fig_feature_ablation.png"))
    plt.close()


def main():
    print("=" * 70)
    print(" 09 — Porównanie końcowe i wykresy")
    print("=" * 70)

    print("\n[1/3] Budowanie tabeli zbiorczej ...")
    summary     = build_summary_table()
    summary_pca = build_summary_table_pca()
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary_table.csv"))
    if summary_pca is not None:
        summary_pca.to_csv(os.path.join(config.RESULTS_DIR,
                                         "summary_table_pca.csv"))
    print(summary[["n_clusters","silhouette","calinski_harabasz",
                   "davies_bouldin","ARI","purity"]].to_string())

    print("\n[2/3] Generowanie wykresów ...")
    plot_kmeans_search();                  print("      ✓ fig_kmeans_search.png")
    plot_gmm_bic();                        print("      ✓ fig_gmm_bic.png")
    plot_silhouette_dual(summary, summary_pca if summary_pca is not None else summary)
    print("      ✓ fig_silhouette_dual_space.png  +  fig_silhouette_compare.png")
    plot_feature_ablation(summary);        print("      ✓ fig_feature_ablation.png")
    plot_internal_metrics(summary);        print("      ✓ fig_internal_metrics.png")
    plot_external_metrics(summary);        print("      ✓ fig_external_metrics.png")
    plot_business("SRS_pct",  "Udział segmentu w przychodach (SRS, %)",
                   "fig_business_srs.png", "SRS [%]");   print("      ✓ fig_business_srs.png")
    plot_business("AOV",      "Średnia wartość zamówienia (AOV)",
                   "fig_business_aov.png", "AOV");       print("      ✓ fig_business_aov.png")
    plot_business("CR_pct",   "Wskaźnik odpływu (Churn Rate, %)",
                   "fig_business_churn.png", "CR [%]");  print("      ✓ fig_business_churn.png")
    plot_business("RaR_pct",  "Revenue at Risk (% przychodu segmentu)",
                   "fig_business_rar.png", "RaR [%]");   print("      ✓ fig_business_rar.png")
    plot_pairwise_ari();                   print("      ✓ fig_pairwise_ari_heatmap.png")

    print("\n[3/3] Gotowe.\n")
    print(f"Wszystkie wyniki w: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
