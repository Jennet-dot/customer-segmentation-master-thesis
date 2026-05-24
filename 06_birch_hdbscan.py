"""
06_birch_hdbscan.py
-------------------
Pipeline: BIRCH (przygotowanie danych) → HDBSCAN (klasteryzacja) — model PORÓWNAWCZY.

Uzasadnienie pipeline'u (sekcje 2.4 i 2.5 pracy):
  Klasyczny HDBSCAN ma złożoność co najmniej O(n log n), ale operuje na pełnej macierzy
  wzajemnej osiągalności i jest wrażliwy na rozmiar zbioru. Dla 86k klientów × 32 wymiary
  zachowuje się stabilniej, gdy dane są wcześniej skompresowane.

  BIRCH (Zhang i in., 1996) realizuje to przez CF-tree:
    - każdy klient zostaje przypisany do najbliższego sub-klastra (liścia drzewa CF),
    - sub-klastry mają zwartą reprezentację (N, LS, SS — wzory 12-14 pracy),
    - po kompresji mamy M << N centroidów sub-klastrów.

  HDBSCAN (Campello i in., 2013) klastruje centroidy sub-klastrów BIRCH:
    - identyfikuje klastry o różnej gęstości i dowolnym kształcie,
    - nie wymaga zadania liczby klastrów (k wynika z gęstości),
    - sub-klastry zaklasyfikowane jako szum dostają etykietę -1.

  Finalna etykieta klienta = etykieta HDBSCAN dla jego sub-klastra BIRCH.

Uruchomienie:
    python 06_birch_hdbscan.py

Wejście:  customers_scaled.pkl
Wyjście:  labels_hdbscan.pkl  — Series: Customer_ID -> nr klastra (-1 = szum)
          birch_hdbscan_info.txt — parametry i statystyki pipeline'u
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, HDBSCAN

import config


def main():
    print("=" * 70)
    print(" 06 — BIRCH → HDBSCAN — model PORÓWNAWCZY")
    print("=" * 70)

    print(f"\n[1/5] Wczytywanie standaryzowanych cech ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    # ----- KROK 1: BIRCH — kompresja danych do sub-klastrów -----
    print(f"\n[2/5] BIRCH: budowa CF-tree "
          f"(threshold={config.BIRCH_THRESHOLD}, branching={config.BIRCH_BRANCHING}) ...")
    birch = Birch(threshold=config.BIRCH_THRESHOLD,
                  branching_factor=config.BIRCH_BRANCHING,
                  n_clusters=config.BIRCH_N_CLUSTERS)
    birch.fit(X)
    centers = birch.subcluster_centers_           # shape (M, D)
    sub_idx = birch.predict(X)                    # shape (N,) — sub-klaster każdego klienta
    n_sub = centers.shape[0]
    print(f"      Utworzono {n_sub:,} sub-klastrów "
          f"(kompresja {X.shape[0] / n_sub:.1f}×)")

    # ----- KROK 2: HDBSCAN na centroidach sub-klastrów -----
    print(f"\n[3/5] HDBSCAN na centroidach sub-klastrów "
          f"(min_cluster_size={config.HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={config.HDBSCAN_MIN_SAMPLES}) ...")
    hdb = HDBSCAN(min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                  min_samples=config.HDBSCAN_MIN_SAMPLES)
    sub_labels = hdb.fit_predict(centers)         # shape (M,) — etykieta każdego sub-klastra

    n_clusters = int((np.unique(sub_labels) >= 0).sum())
    n_noise_sub = int((sub_labels == -1).sum())
    print(f"      HDBSCAN: {n_clusters} klastrów + {n_noise_sub} sub-klastrów oznaczonych jako szum")

    # ----- KROK 3: Propagacja etykiet na klientów -----
    print(f"\n[4/5] Propagacja etykiet HDBSCAN na poziom klientów ...")
    labels_full = sub_labels[sub_idx]            # shape (N,)

    distribution = pd.Series(labels_full).value_counts().sort_index()
    print("      Rozkład klastrów (pełny zbiór klientów):")
    for c, n in distribution.items():
        label_name = f"Szum (-1)" if c == -1 else f"Klaster {c:>2d}"
        print(f"        {label_name:>14s}: {n:>7,}  ({100 * n / len(labels_full):5.2f}%)")

    # ----- KROK 4: Zapis -----
    print("\n[5/5] Zapis etykiet ...")
    pd.Series(labels_full, index=X_df.index, name="HDBSCAN").to_pickle(
        config.PATH_LABELS["HDBSCAN"])

    info_path = os.path.join(config.RESULTS_DIR, "birch_hdbscan_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("BIRCH → HDBSCAN pipeline\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"BIRCH:\n")
        f.write(f"  threshold:        {config.BIRCH_THRESHOLD}\n")
        f.write(f"  branching_factor: {config.BIRCH_BRANCHING}\n")
        f.write(f"  liczba sub-klastrów: {n_sub:,}\n")
        f.write(f"  kompresja: {X.shape[0]} → {n_sub} ({X.shape[0] / n_sub:.1f}×)\n\n")
        f.write(f"HDBSCAN:\n")
        f.write(f"  min_cluster_size: {config.HDBSCAN_MIN_CLUSTER_SIZE}\n")
        f.write(f"  min_samples:      {config.HDBSCAN_MIN_SAMPLES}\n")
        f.write(f"  liczba klastrów:  {n_clusters}\n")
        f.write(f"  sub-klastry-szum: {n_noise_sub}\n\n")
        f.write(f"Wynik na poziomie klientów ({len(labels_full):,}):\n")
        for c, n in distribution.items():
            label_name = "Szum (-1)" if c == -1 else f"Klaster {c}"
            f.write(f"  {label_name:>14s}: {n:>7,} "
                    f"({100 * n / len(labels_full):5.2f}%)\n")

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['HDBSCAN'])} "
          f"({n_clusters} klastrów + szum)")
    print(f"      ✓ results/birch_hdbscan_info.txt")
    print("\n[OK] BIRCH → HDBSCAN zakończony.\n")


if __name__ == "__main__":
    main()
