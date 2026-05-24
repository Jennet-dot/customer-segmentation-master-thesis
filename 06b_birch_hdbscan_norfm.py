"""
06b_birch_hdbscan_norfm.py
--------------------------
Pipeline BIRCH → HDBSCAN na cechach BEZ wskaźników RFM — wariant do testu hipotezy.

Metoda taka sama jak w 06_birch_hdbscan.py:
  1) BIRCH kompresuje N obserwacji do M sub-klastrów.
  2) HDBSCAN klastruje centroidy sub-klastrów.
  3) Klient dziedziczy etykietę swojego sub-klastra.

UWAGA: BIRCH_THRESHOLD jest skalibrowany dla zbioru 32-wymiarowego pełnego.
Dla 26-wymiarowego zbioru no_rfm typowe odległości euklidesowe są mniejsze
(brak silnie zróżnicowanych cech Monetary), więc threshold dobieramy nieco
niżej w trakcie auto-strojenia.

Uruchomienie:
    python 06b_birch_hdbscan_norfm.py

Wejście:  customers_scaled_norfm.pkl
Wyjście:  labels_hdbscan_norfm.pkl
          birch_hdbscan_norfm_info.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, HDBSCAN

import config


def autotune_birch_threshold(X: np.ndarray,
                              target_n_subclusters_range=(500, 3000),
                              start_threshold: float = 4.0) -> float:
    """
    Dobiera próg BIRCH tak, by liczba sub-klastrów mieściła się w docelowym przedziale.
    Dane mają inne rozkłady niż pełny zbiór, więc threshold=4.0 nie musi pasować.
    """
    lo, hi = target_n_subclusters_range
    t = start_threshold
    # Najpierw szybki test z initialnym thresholdem
    b = Birch(threshold=t, branching_factor=config.BIRCH_BRANCHING, n_clusters=None)
    b.fit(X)
    n = b.subcluster_centers_.shape[0]
    print(f"      próba t={t:.2f}: {n:,} sub-klastrów")
    if lo <= n <= hi:
        return t

    # Wyszukiwanie binarne
    t_min, t_max = 0.5, 10.0
    for _ in range(15):
        t = (t_min + t_max) / 2
        b = Birch(threshold=t, branching_factor=config.BIRCH_BRANCHING, n_clusters=None)
        b.fit(X)
        n = b.subcluster_centers_.shape[0]
        print(f"      próba t={t:.2f}: {n:,} sub-klastrów")
        if lo <= n <= hi:
            return t
        if n > hi:
            t_min = t  # za dużo → zwiększamy próg
        else:
            t_max = t  # za mało → zmniejszamy próg
    return t


def main():
    print("=" * 70)
    print(" 06b — BIRCH → HDBSCAN BEZ cech RFM — test hipotezy badawczej")
    print("=" * 70)

    print(f"\n[1/5] Wczytywanie cech BEZ RFM ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    print(f"\n[2/5] Auto-tuning progu BIRCH (cel: 500–3000 sub-klastrów) ...")
    threshold = autotune_birch_threshold(X)
    print(f"      Wybrany threshold: {threshold:.2f}")

    print(f"\n[3/5] BIRCH: budowa CF-tree ...")
    birch = Birch(threshold=threshold,
                  branching_factor=config.BIRCH_BRANCHING,
                  n_clusters=config.BIRCH_N_CLUSTERS)
    birch.fit(X)
    centers = birch.subcluster_centers_
    sub_idx = birch.predict(X)
    n_sub = centers.shape[0]
    print(f"      Sub-klastrów: {n_sub:,} (kompresja {X.shape[0] / n_sub:.1f}×)")

    print(f"\n[4/5] HDBSCAN na centroidach "
          f"(min_cluster_size={config.HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={config.HDBSCAN_MIN_SAMPLES}) ...")
    hdb = HDBSCAN(min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                  min_samples=config.HDBSCAN_MIN_SAMPLES)
    sub_labels = hdb.fit_predict(centers)
    n_clusters = int((np.unique(sub_labels) >= 0).sum())
    n_noise_sub = int((sub_labels == -1).sum())
    print(f"      HDBSCAN: {n_clusters} klastrów + {n_noise_sub} sub-klastrów-szumu")

    labels_full = sub_labels[sub_idx]
    distribution = pd.Series(labels_full).value_counts().sort_index()
    print("\n      Rozkład klastrów (pełny zbiór):")
    for c, n in distribution.items():
        name = "Szum (-1)" if c == -1 else f"Klaster {c:>2d}"
        print(f"        {name:>14s}: {n:>7,}  ({100 * n / len(labels_full):5.2f}%)")

    print("\n[5/5] Zapis etykiet ...")
    pd.Series(labels_full, index=X_df.index, name="HDBSCAN_noRFM").to_pickle(
        config.PATH_LABELS["HDBSCAN_noRFM"])

    info_path = os.path.join(config.RESULTS_DIR, "birch_hdbscan_norfm_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("BIRCH → HDBSCAN BEZ cech RFM\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"BIRCH:\n")
        f.write(f"  threshold (auto): {threshold:.3f}\n")
        f.write(f"  branching_factor: {config.BIRCH_BRANCHING}\n")
        f.write(f"  liczba sub-klastrów: {n_sub:,}\n\n")
        f.write(f"HDBSCAN:\n")
        f.write(f"  liczba klastrów: {n_clusters}\n")
        f.write(f"  sub-klastry-szum: {n_noise_sub}\n\n")
        f.write(f"Wynik:\n")
        for c, n in distribution.items():
            name = "Szum (-1)" if c == -1 else f"Klaster {c}"
            f.write(f"  {name:>14s}: {n:>7,} "
                    f"({100 * n / len(labels_full):5.2f}%)\n")

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['HDBSCAN_noRFM'])}")
    print(f"      ✓ results/birch_hdbscan_norfm_info.txt")
    print("\n[OK] BIRCH → HDBSCAN bez RFM zakończony.\n")


if __name__ == "__main__":
    main()
