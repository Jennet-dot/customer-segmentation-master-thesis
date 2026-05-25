"""
12_dendrogram.py
----------------
Wizualizacja hierarchicznej struktury klastrów algorytmu Warda za pomocą dendrogramu.

Dendrogram pokazuje proces aglomeracji: jak kolejne pary klastrów łączą się
w coraz większe grupy oraz na jakiej "wysokości" (odległości Warda) zachodzi
każde łączenie. Pozwala to:
  (1) Wizualnie zweryfikować wybraną liczbę klastrów k = 5.
  (2) Zobaczyć, które klastry są sobie bliższe (łączą się niżej), a które
      dalsze (łączą się wyżej).
  (3) Zaobserwować, czy struktura jest zrównoważona, czy jeden klaster
      dominuje nad innymi.

Generujemy DWA dendrogramy:
  - fig_dendrogram_truncated.png — ostatnie 30 łączeń z 10 000 (czytelny,
    pokazuje strukturę wysokopoziomową, najczęściej używany w pracach).
  - fig_dendrogram_full_subsample.png — pełny dendrogram dla podpróbki
    300 klientów (mniejsza próbka, ale pokazuje strukturę "od dołu").

Uruchomienie:
    python 12_dendrogram.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

import config

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def build_dendrogram_truncated(X: np.ndarray, k: int,
                                output_path: str,
                                n_clusters_to_show: int = 30,
                                title_suffix: str = ""):
    """
    Wizualizuje OSTATNIE n_clusters_to_show łączeń w dendrogramie Warda.

    Używa trybu truncate_mode='lastp' z scipy — pokazuje tylko ostatnie p
    łączeń, agregując niższe poziomy. Czytelny i informacyjny.
    """
    print(f"      Liczenie linkage Warda dla {X.shape[0]:,} punktów ...")
    Z = linkage(X, method="ward")

    # Oblicz progi dla wybranego k — gdzie odciąć dendrogram poziomo
    # Wysokości złączeń są w trzeciej kolumnie Z
    heights = sorted(Z[:, 2], reverse=True)
    # Dla k klastrów odcinamy nad k-tym największym złączeniem
    cut_height = (heights[k - 2] + heights[k - 1]) / 2 if k >= 2 else heights[0]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Kolory — przygotuj k kolorów dla klastrów
    color_map = matplotlib.colormaps["tab10"]
    cluster_colors = [matplotlib.colors.rgb2hex(color_map(i)) for i in range(k)]

    # Dendrogram z color_threshold = cut_height
    ddata = dendrogram(
        Z,
        truncate_mode="lastp",
        p=n_clusters_to_show,
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=cut_height,
        above_threshold_color="gray",
        ax=ax,
    )

    # Linia odcięcia dla k klastrów
    ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.5,
                label=f"Próg dla k = {k} (wys. ≈ {cut_height:.2f})")

    ax.set_title(f"Dendrogram aglomeracji Warda — ostatnich "
                  f"{n_clusters_to_show} łączeń{title_suffix}",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Klastry (w nawiasach — liczba obserwacji)")
    ax.set_ylabel("Odległość Warda (ESS)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_dendrogram_full_subsample(X: np.ndarray, k: int,
                                      output_path: str,
                                      subsample_size: int = 300,
                                      random_state: int = 42,
                                      title_suffix: str = ""):
    """
    Wizualizuje PEŁNY dendrogram dla mniejszej podpróbki (np. 300 klientów).

    Mniejsza próbka pozwala pokazać każde indywidualne łączenie i daje
    intuicję, jak struktura buduje się od dołu.
    """
    rng = np.random.default_rng(random_state)
    if X.shape[0] > subsample_size:
        idx = rng.choice(X.shape[0], subsample_size, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    print(f"      Liczenie linkage Warda dla {X_sub.shape[0]} punktów podpróbki ...")
    Z = linkage(X_sub, method="ward")

    heights = sorted(Z[:, 2], reverse=True)
    cut_height = (heights[k - 2] + heights[k - 1]) / 2 if k >= 2 else heights[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        Z,
        leaf_rotation=90,
        leaf_font_size=4,
        color_threshold=cut_height,
        above_threshold_color="gray",
        no_labels=True,  # przy 300 obiektach etykiety są nieczytelne
        ax=ax,
    )

    ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.5,
                label=f"Próg dla k = {k} (wys. ≈ {cut_height:.2f})")

    ax.set_title(f"Pełny dendrogram aglomeracji Warda — "
                  f"podpróbka {X_sub.shape[0]} klientów{title_suffix}",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Indywidualni klienci (n = {X_sub.shape[0]})")
    ax.set_ylabel("Odległość Warda (ESS)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    print("=" * 70)
    print(" 12 — Dendrogramy klasteryzacji hierarchicznej Warda")
    print("=" * 70)

    print("\n[1/3] Wczytywanie danych ...")
    customers = pd.read_pickle(config.PATH_CUSTOMERS)
    X_scaled = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X_scaled_norfm = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)

    rfm_cols = ["Recency", "Frequency", "Monetary"]
    X_rfm_3d = StandardScaler().fit_transform(
        customers.loc[X_scaled.index, rfm_cols].values)

    rng = np.random.default_rng(config.RANDOM_STATE)

    # Próbka 10k (taka jak w 05_hierarchical.py)
    n_sample = 10000
    if X_scaled.shape[0] > n_sample:
        sample_idx = rng.choice(X_scaled.shape[0], n_sample, replace=False)
        X_full_sample  = X_scaled.values[sample_idx]
        X_norfm_sample = X_scaled_norfm.values[sample_idx]
        X_rfm_sample   = X_rfm_3d[sample_idx]
    else:
        X_full_sample, X_norfm_sample, X_rfm_sample = (
            X_scaled.values, X_scaled_norfm.values, X_rfm_3d)

    print(f"      Wykorzystana próbka: {len(X_full_sample):,} klientów\n")

    # ----- Dendrogramy dla modelu Hierarchical (pełna przestrzeń 32D) -----
    print("[2/3] Generowanie dendrogramów dla wariantu pełnego (32D) ...\n")

    print("      → Dendrogram skrócony (ostatnich 30 łączeń):")
    build_dendrogram_truncated(
        X=X_full_sample, k=5,
        output_path=os.path.join(config.RESULTS_DIR,
                                   "fig_dendrogram_truncated.png"),
        n_clusters_to_show=30,
        title_suffix=" (model Hierarchical, przestrzeń 32D)")
    print(f"        ✓ fig_dendrogram_truncated.png")

    print("\n      → Dendrogram pełny dla podpróbki 300 klientów:")
    build_dendrogram_full_subsample(
        X=X_full_sample, k=5,
        output_path=os.path.join(config.RESULTS_DIR,
                                   "fig_dendrogram_full_subsample.png"),
        subsample_size=300,
        random_state=config.RANDOM_STATE,
        title_suffix=" (model Hierarchical, przestrzeń 32D)")
    print(f"        ✓ fig_dendrogram_full_subsample.png")

    # ----- Dendrogram dla wariantu noRFM (26D) -----
    print("\n[3/3] Dendrogram skrócony dla wariantu noRFM (26D) ...")
    build_dendrogram_truncated(
        X=X_norfm_sample, k=4,
        output_path=os.path.join(config.RESULTS_DIR,
                                   "fig_dendrogram_truncated_noRFM.png"),
        n_clusters_to_show=30,
        title_suffix=" (model Hierarchical_noRFM, przestrzeń 26D)")
    print(f"        ✓ fig_dendrogram_truncated_noRFM.png")

    print(f"\n[OK] Dendrogramy zapisane w: {config.RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
