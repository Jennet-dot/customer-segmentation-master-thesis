"""
07_gmm.py
---------
Gaussian Mixture Models (Bishop, 2006) — model PORÓWNAWCZY.

Metoda (sekcja 2.5 pracy, wzory 28-38):
    p(x) = sum_{k=1..K} pi_k * N(x | mu_k, Sigma_k)
gdzie:
    pi_k                 — współczynniki mieszania (sum = 1, wzór 29)
    mu_k, Sigma_k        — średnia i macierz kowariancji k-tego komponentu
    N(x | mu_k, Sigma_k) — wielowymiarowy rozkład normalny

Parametry estymowane algorytmem Expectation-Maximization (Dempster i in., 1977):
    Krok E: gamma(z_nk) — odpowiedzialność komponentu (wzór 33)
    Krok M: aktualizacja mu, Sigma, pi (wzory 35-37)

Dobór K (liczby komponentów):
  Stosujemy kryterium BIC (Bayesian Information Criterion) — niższy BIC = lepszy model.
  BIC karze złożoność modelu, dzięki czemu dobrze radzi sobie z przypadkiem GMM, gdzie
  log-likelihood zawsze rośnie z K. Dodatkowo liczymy silhouette (na próbce) dla
  spójności z pozostałymi modelami.

W przeciwieństwie do K-means GMM przypisuje obserwacjom PRAWDOPODOBIEŃSTWA przynależności
do każdego komponentu (soft clustering). Dla porównania z innymi modelami zapisujemy
twarde etykiety = argmax_k gamma(z_nk).

Uruchomienie:
    python 07_gmm.py

Wejście:  customers_scaled.pkl
Wyjście:  labels_gmm.pkl       — Series: Customer_ID -> nr komponentu (hard assignment)
          gmm_search.csv       — K, BIC, AIC, silhouette
          gmm_responsibilities.pkl — DataFrame z prawdopodobieństwami (do dalszych analiz)
"""

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import config


def search_optimal_k_gmm(X: np.ndarray, k_range: list, random_state: int,
                          silhouette_sample: int = 10000) -> pd.DataFrame:
    """Dla każdego K liczy BIC, AIC i silhouette."""
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
    print(" 07 — Gaussian Mixture Models (Bishop, 2006) — model PORÓWNAWCZY")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie standaryzowanych cech ...")
    X_df = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X = X_df.values
    print(f"      Klienci: {X.shape[0]:,}, cechy: {X.shape[1]}")

    print(f"\n[2/4] Wyszukiwanie optymalnego K z zakresu "
          f"{config.K_RANGE[0]}..{config.K_RANGE[-1]} (kryterium BIC) ...")
    search = search_optimal_k_gmm(X, config.K_RANGE, config.RANDOM_STATE)
    best_k = int(search.loc[search["bic"].idxmin(), "k"])
    print(f"\n      Najlepsze K wg BIC: K={best_k} "
          f"(BIC={search['bic'].min():,.2f}, "
          f"silhouette dla tego K: {search.loc[search['k'] == best_k, 'silhouette'].iloc[0]:.4f})")

    print(f"\n[3/4] Trenowanie finalnego GMM dla K={best_k} (covariance_type='full', "
          f"algorytm EM) ...")
    gmm_final = GaussianMixture(n_components=best_k, covariance_type="full",
                                max_iter=500, n_init=10,
                                random_state=config.RANDOM_STATE, reg_covar=1e-4)
    gmm_final.fit(X)

    # Hard assignment (argmax z prawdopodobieństw a posteriori)
    labels = gmm_final.predict(X)
    # Soft assignment (responsibilities) — do dalszej analizy
    responsibilities = gmm_final.predict_proba(X)

    distribution = pd.Series(labels).value_counts().sort_index()
    print("      Rozkład komponentów (hard assignment = argmax responsibility):")
    for c, n in distribution.items():
        print(f"        Komponent {c:>2d}: {n:>7,}  ({100 * n / len(labels):5.2f}%)  "
              f"pi_{c}={gmm_final.weights_[c]:.4f}")

    print("\n[4/4] Zapis wyników ...")
    pd.Series(labels, index=X_df.index, name="GMM").to_pickle(config.PATH_LABELS["GMM"])
    pd.DataFrame(responsibilities, index=X_df.index,
                 columns=[f"P(C{k})" for k in range(best_k)]).to_pickle(
        os.path.join(config.DATA_DIR, "gmm_responsibilities.pkl"))
    search.to_csv(os.path.join(config.RESULTS_DIR, "gmm_search.csv"), index=False)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['GMM'])} "
          f"(K={best_k} komponentów)")
    print(f"      ✓ data/gmm_responsibilities.pkl")
    print(f"      ✓ results/gmm_search.csv")
    print("\n[OK] GMM zakończony.\n")


if __name__ == "__main__":
    main()
