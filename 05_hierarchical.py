"""
05_hierarchical.py
------------------
Klasteryzacja hierarchiczna aglomeracyjna (AGNES) — model PORÓWNAWCZY.

Metoda (sekcja 2.4 pracy):
  - Łączenie Warda (Ward, 1963) — minimalizacja przyrostu SSE przy łączeniu klastrów.
    Wybrane jako metoda główna spośród pięciu wymienionych w pracy
    (single, complete, average, centroid, Ward); Ward najczęściej cytowana w literaturze
    e-commerce (Han, Kamber, Pei, 2011, str. 500-506).

Problem skalowalności (sekcja 2.4 pracy, akapit o ograniczeniach):
  Pełna macierz odległości dla n=86,691 wymagałaby ~30 GB pamięci.
  Rozwiązanie (zgodne z duchem BIRCH wzmiankowanego w pracy):
    1) Próbkujemy HIER_SAMPLE_SIZE klientów (domyślnie 10,000).
    2) Trenujemy AgglomerativeClustering(linkage='ward') na próbce.
    3) Liczymy centroidy klastrów uzyskanych na próbce.
    4) Pozostałych klientów przypisujemy do najbliższego centroidu (1-NN).
  Próbka 10k klientów jest reprezentatywna (~12% populacji) i stabilna statystycznie.

Dobór k:
  Silhouette dla k = 2..10 na próbce. Argmax wybiera finalne k.

Uruchomienie:
    python 05_hierarchical.py

Wejście:  customers_scaled.pkl
Wyjście:  labels_hierarchical.pkl  — Series: Customer_ID -> nr klastra
          hierarchical_search.csv  — k, silhouette
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
    """Liczy silhouette dla każdego k na próbce."""
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
    print(" 05 — Hierarchical (Ward) — model PORÓWNAWCZY")
    print("=" * 70)

    print(f"\n[1/5] Wczytywanie standaryzowanych cech ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    # ----- Próbka reprezentatywna -----
    n_sample = min(config.HIER_SAMPLE_SIZE, X.shape[0])
    print(f"\n[2/5] Próbkowanie {n_sample:,} klientów "
          f"(pełny dataset niewykonalny pamięciowo dla Warda) ...")
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
    X_sample = X[sample_idx]

    # ----- Dobór k -----
    print(f"\n[3/5] Wyszukiwanie optymalnego k (linkage='{config.HIER_LINKAGE}') ...")
    search = search_optimal_k_hierarchical(X_sample, config.K_RANGE,
                                           linkage=config.HIER_LINKAGE)
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    print(f"\n      Najlepsze k wg silhouette: k={best_k} "
          f"(sil={search['silhouette'].max():.4f})")

    # ----- Finalny model na próbce -----
    print(f"\n[4/5] Finalna hierarchia (Ward) dla k={best_k} na próbce {n_sample:,} ...")
    final_model = AgglomerativeClustering(n_clusters=best_k,
                                          linkage=config.HIER_LINKAGE)
    labels_sample = final_model.fit_predict(X_sample)

    # ----- Propagacja etykiet do pełnego zbioru: 1-NN do centroidów -----
    print(f"      Propagacja etykiet na pozostałych "
          f"{X.shape[0] - n_sample:,} klientów (1-NN do centroidów) ...")
    nc = NearestCentroid()
    nc.fit(X_sample, labels_sample)
    labels_full = nc.predict(X)

    # Spójność: dla klientów z próbki ustawiamy znane etykiety (nawet jeśli ich centroid
    # różniłby się minimalnie od wyniku 1-NN)
    labels_full[sample_idx] = labels_sample

    distribution = pd.Series(labels_full).value_counts().sort_index()
    print("\n      Rozkład klastrów (pełny zbiór):")
    for c, n in distribution.items():
        print(f"        Klaster {c:>2d}: {n:>7,}  ({100 * n / len(labels_full):5.2f}%)")

    # ----- Zapis -----
    print("\n[5/5] Zapis etykiet i wyników poszukiwania ...")
    pd.Series(labels_full, index=X_df.index, name="Hierarchical").to_pickle(
        config.PATH_LABELS["Hierarchical"])
    search.to_csv(os.path.join(config.RESULTS_DIR, "hierarchical_search.csv"), index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['Hierarchical'])} "
          f"(k={best_k} klastrów)")
    print(f"      ✓ results/hierarchical_search.csv")
    print("\n[OK] Hierarchical zakończony.\n")


if __name__ == "__main__":
    main()
