"""
04_kmeans.py
------------
K-średnich (MacQueen, 1967) — model PORÓWNAWCZY.

Metoda (sekcja 2.3 pracy, wzór 11):
    J = sum_{i=1..k} sum_{x_j in C_i} ||x_j - mu_i||^2

Algorytm minimalizuje wewnątrzklastrową sumę kwadratów (WCSS / inercję).
Inicjalizacja: k-means++ (Arthur i Vassilvitskii, 2007 — wzmiankowane w pracy).

Dobór k:
  - Liczymy inercję i silhouette dla k = 2..10.
  - "Optymalne" k = argmax silhouette (bardziej obiektywne niż wzrokowa metoda łokcia,
    która nadal jest pokazywana na wykresie do tezy).

Wszystkie cechy z preprocessing.full (32 cechy: numeryczne + porządkowe + one-hot),
po standaryzacji z-score (wzory 3-6 z pracy).

Uruchomienie:
    python 04_kmeans.py

Wejście:  customers_scaled.pkl
Wyjście:  labels_kmeans.pkl  — Series: Customer_ID -> nr klastra
          kmeans_search.csv  — k, inercja, silhouette (do wykresów w pracy)
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config


def search_optimal_k(X: np.ndarray, k_range: list, random_state: int,
                     silhouette_sample: int = 10000) -> pd.DataFrame:
    """
    Dla każdego k w k_range trenuje KMeans i liczy:
      - inertia_ (do metody łokcia)
      - silhouette_score (na próbce, bo pełne obliczenia O(n^2))
    Zwraca DataFrame z kolumnami: k, inertia, silhouette.
    """
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
    print(" 04 — K-means (MacQueen, 1967) — model PORÓWNAWCZY")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie standaryzowanych cech z {config.PATH_CUSTOMERS_SCALED} ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    print(f"\n[2/4] Wyszukiwanie optymalnego k z zakresu {config.K_RANGE[0]}..{config.K_RANGE[-1]} ...")
    print("      (inertia → metoda łokcia, silhouette → wybór finalny)")
    search = search_optimal_k(X, config.K_RANGE, config.RANDOM_STATE)

    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    print(f"\n      Najlepsze k wg silhouette: k={best_k} "
          f"(sil={search['silhouette'].max():.4f})")

    print(f"\n[3/4] Trenowanie finalnego modelu K-means dla k={best_k} ...")
    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20,
                      random_state=config.RANDOM_STATE)
    labels = km_final.fit_predict(X)

    distribution = pd.Series(labels).value_counts().sort_index()
    print("      Rozkład klastrów:")
    for c, n in distribution.items():
        print(f"        Klaster {c:>2d}: {n:>7,}  ({100 * n / len(labels):5.2f}%)")

    print("\n[4/4] Zapis etykiet i wyników poszukiwania ...")
    pd.Series(labels, index=X_df.index, name="KMeans").to_pickle(
        config.PATH_LABELS["KMeans"])
    search.to_csv(os.path.join(config.RESULTS_DIR, "kmeans_search.csv"), index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['KMeans'])} "
          f"(k={best_k} klastrów)")
    print(f"      ✓ results/kmeans_search.csv  (do wykresów inertia/silhouette)")
    print("\n[OK] K-means zakończony.\n")


if __name__ == "__main__":
    main()
