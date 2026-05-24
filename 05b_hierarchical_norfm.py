"""
05b_hierarchical_norfm.py
-------------------------
Klasteryzacja hierarchiczna Warda na cechach BEZ wskaźników RFM — wariant do testu
hipotezy badawczej.

Cel: jak w 04b_kmeans_norfm.py — sprawdzenie, czy hierarchical Ward odkryje
strukturę segmentów wyłącznie na podstawie cech demograficzno-behawioralnych.

Metoda taka sama jak w 05_hierarchical.py:
  - próbka HIER_SAMPLE_SIZE klientów,
  - AgglomerativeClustering(linkage='ward') na próbce,
  - propagacja etykiet do pełnego zbioru przez 1-NN do centroidów.

Uruchomienie:
    python 05b_hierarchical_norfm.py

Wejście:  customers_scaled_norfm.pkl
Wyjście:  labels_hierarchical_norfm.pkl
          hierarchical_norfm_search.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid

import config


def search_optimal_k_hierarchical(X_sample: np.ndarray, k_range: list,
                                  linkage: str) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        rows.append({"k": k, "silhouette": float(sil)})
        print(f"      k={k:2d}   silhouette={sil:.4f}")
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print(" 05b — Hierarchical (Ward) BEZ cech RFM — test hipotezy badawczej")
    print("=" * 70)

    print(f"\n[1/5] Wczytywanie cech BEZ RFM ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    n_sample = min(config.HIER_SAMPLE_SIZE, X.shape[0])
    print(f"\n[2/5] Próbkowanie {n_sample:,} klientów ...")
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
    X_sample = X[sample_idx]

    print(f"\n[3/5] Wyszukiwanie optymalnego k (linkage='{config.HIER_LINKAGE}') ...")
    search = search_optimal_k_hierarchical(X_sample, config.K_RANGE,
                                           linkage=config.HIER_LINKAGE)
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    print(f"\n      Najlepsze k: k={best_k} (sil={search['silhouette'].max():.4f})")

    print(f"\n[4/5] Finalna hierarchia (Ward) dla k={best_k} ...")
    final_model = AgglomerativeClustering(n_clusters=best_k,
                                          linkage=config.HIER_LINKAGE)
    labels_sample = final_model.fit_predict(X_sample)

    print(f"      Propagacja etykiet (1-NN do centroidów) ...")
    nc = NearestCentroid()
    nc.fit(X_sample, labels_sample)
    labels_full = nc.predict(X)
    labels_full[sample_idx] = labels_sample

    distribution = pd.Series(labels_full).value_counts().sort_index()
    print("\n      Rozkład klastrów:")
    for c, n in distribution.items():
        print(f"        Klaster {c:>2d}: {n:>7,}  ({100 * n / len(labels_full):5.2f}%)")

    print("\n[5/5] Zapis ...")
    pd.Series(labels_full, index=X_df.index,
              name="Hierarchical_noRFM").to_pickle(
        config.PATH_LABELS["Hierarchical_noRFM"])
    search.to_csv(os.path.join(config.RESULTS_DIR, "hierarchical_norfm_search.csv"),
                  index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['Hierarchical_noRFM'])}")
    print(f"      ✓ results/hierarchical_norfm_search.csv")
    print("\n[OK] Hierarchical bez RFM zakończony.\n")


if __name__ == "__main__":
    main()
