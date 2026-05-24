"""
07b_gmm_norfm.py
----------------
Gaussian Mixture Models na cechach BEZ wskaźników RFM — wariant do testu hipotezy.

Cel: jak w pozostałych wariantach _noRFM. Sprawdzenie, czy soft clustering GMM
identyfikuje sensowną strukturę gęstości w przestrzeni cech demograficzno-
behawioralnych bez RFM.

Uruchomienie:
    python 07b_gmm_norfm.py

Wejście:  customers_scaled_norfm.pkl
Wyjście:  labels_gmm_norfm.pkl
          gmm_norfm_search.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import config


def search_optimal_k_gmm(X: np.ndarray, k_range: list, random_state: int,
                          silhouette_sample: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if X.shape[0] > silhouette_sample:
        sample_idx = rng.choice(X.shape[0], size=silhouette_sample, replace=False)
    else:
        sample_idx = np.arange(X.shape[0])
    X_sil = X[sample_idx]

    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              max_iter=200, n_init=3,
                              random_state=random_state, reg_covar=1e-4)
        gmm.fit(X)
        labels = gmm.predict(X)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        if len(np.unique(labels[sample_idx])) > 1:
            sil = float(silhouette_score(X_sil, labels[sample_idx]))
        else:
            sil = float("nan")
        rows.append({"k": k, "bic": bic, "aic": aic, "silhouette": sil})
        print(f"      K={k:2d}   BIC={bic:>14,.2f}   AIC={aic:>14,.2f}   silhouette={sil:.4f}")
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print(" 07b — Gaussian Mixture Models BEZ cech RFM — test hipotezy")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie cech BEZ RFM ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    print(f"\n[2/4] Wyszukiwanie optymalnego K (kryterium BIC) ...")
    search = search_optimal_k_gmm(X, config.K_RANGE, config.RANDOM_STATE)
    best_k = int(search.loc[search["bic"].idxmin(), "k"])
    print(f"\n      Najlepsze K wg BIC: K={best_k} "
          f"(BIC={search['bic'].min():,.2f})")

    print(f"\n[3/4] Trenowanie finalnego GMM dla K={best_k} ...")
    gmm_final = GaussianMixture(n_components=best_k, covariance_type="full",
                                max_iter=500, n_init=10,
                                random_state=config.RANDOM_STATE, reg_covar=1e-4)
    gmm_final.fit(X)
    labels = gmm_final.predict(X)

    distribution = pd.Series(labels).value_counts().sort_index()
    print("      Rozkład komponentów:")
    for c, n in distribution.items():
        print(f"        Komponent {c:>2d}: {n:>7,}  ({100 * n / len(labels):5.2f}%)  "
              f"pi_{c}={gmm_final.weights_[c]:.4f}")

    print("\n[4/4] Zapis ...")
    pd.Series(labels, index=X_df.index, name="GMM_noRFM").to_pickle(
        config.PATH_LABELS["GMM_noRFM"])
    search.to_csv(os.path.join(config.RESULTS_DIR, "gmm_norfm_search.csv"),
                  index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['GMM_noRFM'])} (K={best_k})")
    print(f"      ✓ results/gmm_norfm_search.csv")
    print("\n[OK] GMM bez RFM zakończony.\n")


if __name__ == "__main__":
    main()
