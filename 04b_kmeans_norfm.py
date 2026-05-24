"""
04b_kmeans_norfm.py
-------------------
K-średnich na cechach BEZ wskaźników RFM — wariant do testu hipotezy badawczej.

Cel:
  Sprawdzić, czy zaawansowany algorytm klasteryzacji (K-means) potrafi odkryć
  sensowną strukturę segmentów wykorzystując WYŁĄCZNIE cechy demograficzno-
  behawioralne klienta (wiek, dochód, kraj, kategoria produktu, sposób płatności,
  dostawa, ocena, itp.), bez bezpośredniego dostępu do wskaźników Recency,
  Frequency i Monetary.

  Porównanie wyników tego wariantu z modelem bazowym RFM oraz z modelem K-means
  na pełnym zestawie cech pozwala odpowiedzieć na pytania:
    (1) czy cechy demograficzno-behawioralne są wystarczające do segmentacji?
    (2) ile do segmentacji wnosi sama informacja R/F/M, a ile inne cechy?

Uruchomienie:
    python 04b_kmeans_norfm.py

Wejście:  customers_scaled_norfm.pkl
Wyjście:  labels_kmeans_norfm.pkl  — Series: Customer_ID -> nr klastra
          kmeans_norfm_search.csv  — k, inercja, silhouette
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config


def search_optimal_k(X: np.ndarray, k_range: list, random_state: int,
                     silhouette_sample: int = 10000) -> pd.DataFrame:
    """Dla każdego k trenuje KMeans i liczy inercję + silhouette."""
    rng = np.random.default_rng(random_state)
    if X.shape[0] > silhouette_sample:
        sample_idx = rng.choice(X.shape[0], size=silhouette_sample, replace=False)
    else:
        sample_idx = np.arange(X.shape[0])
    X_sil = X[sample_idx]

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X_sil, labels[sample_idx])
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
        print(f"      k={k:2d}   inertia={km.inertia_:>14,.2f}   silhouette={sil:.4f}")
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print(" 04b — K-means BEZ cech RFM — test hipotezy badawczej")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie cech BEZ RFM z {config.PATH_CUSTOMERS_SCALED_NORFM} ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]} (bez Recency/Frequency/Monetary)")

    print(f"\n[2/4] Wyszukiwanie optymalnego k z zakresu "
          f"{config.K_RANGE[0]}..{config.K_RANGE[-1]} ...")
    search = search_optimal_k(X, config.K_RANGE, config.RANDOM_STATE)
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    print(f"\n      Najlepsze k wg silhouette: k={best_k} "
          f"(sil={search['silhouette'].max():.4f})")

    print(f"\n[3/4] Trenowanie finalnego K-means dla k={best_k} ...")
    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20,
                      random_state=config.RANDOM_STATE)
    labels = km_final.fit_predict(X)

    distribution = pd.Series(labels).value_counts().sort_index()
    print("      Rozkład klastrów:")
    for c, n in distribution.items():
        print(f"        Klaster {c:>2d}: {n:>7,}  ({100 * n / len(labels):5.2f}%)")

    print("\n[4/4] Zapis etykiet i wyników poszukiwania ...")
    pd.Series(labels, index=X_df.index, name="KMeans_noRFM").to_pickle(
        config.PATH_LABELS["KMeans_noRFM"])
    search.to_csv(os.path.join(config.RESULTS_DIR, "kmeans_norfm_search.csv"),
                  index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['KMeans_noRFM'])} "
          f"(k={best_k} klastrów)")
    print(f"      ✓ results/kmeans_norfm_search.csv")
    print("\n[OK] K-means bez RFM zakończony.\n")


if __name__ == "__main__":
    main()
