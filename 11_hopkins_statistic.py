"""
11_hopkins_statistic.py
-----------------------
Wyznaczanie statystyki Hopkinsa (Hopkins i Skellam, 1954) — miernika tendencji
klastrowej w danych.

Statystyka Hopkinsa odpowiada na fundamentalne pytanie: czy w danych w ogóle
istnieje struktura klastrowa, czy też są one rozłożone losowo? Interpretacja:
  - H ≈ 0,5  → dane losowe, brak tendencji klastrowej
  - H > 0,75 → silna tendencja klastrowa
  - H ≈ 1,0  → dane mają bardzo wyraźną strukturę klastrową

Definicja (formalna):
    H = sum(w_i) / (sum(u_i) + sum(w_i))
gdzie:
    u_i — odległość losowej obserwacji z danych do najbliższego sąsiada,
    w_i — odległość losowego punktu z przestrzeni cech do najbliższej obserwacji.

Statystyka Hopkinsa NIE zależy od żadnego algorytmu klasteryzacji — ocenia ona
SAME DANE. Niskie wartości silhouette dla modeli mogą wynikać albo z braku
struktury w danych (wtedy H byłoby ~0,5), albo z ograniczeń miernika silhouette
w wysokowymiarowej przestrzeni (curse of dimensionality). Statystyka Hopkinsa
pozwala rozróżnić te dwie sytuacje.

Uruchomienie:
    python 11_hopkins_statistic.py

Wejście:  customers.pkl, customers_scaled.pkl, customers_scaled_norfm.pkl
Wyjście:  results/hopkins_statistic.csv  — tabela H dla każdej przestrzeni
          results/fig_hopkins.png         — wizualizacja
          results/distance_stats.csv      — pomocnicze: CoV odległości
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

import config

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


def hopkins_statistic(X: np.ndarray, n_samples: int = 500,
                      n_repeats: int = 10, random_state: int = 42) -> dict:
    """
    Wyznacza statystykę Hopkinsa metodą Monte Carlo.

    Powtarzamy obliczenie n_repeats razy z różnymi próbkami losowymi
    i zwracamy średnią + odchylenie standardowe — daje to bardziej
    wiarygodną wartość niż pojedynczy pomiar.

    Parametry:
        X            — macierz danych (n × d), powinna być standaryzowana
        n_samples    — liczba punktów próbki (m); literatura zaleca m << n
        n_repeats    — liczba niezależnych powtórzeń
        random_state — ziarno losowości

    Zwraca:
        dict z kluczami: H_mean, H_std, n, d, n_samples
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    n_samples = min(n_samples, n // 2)

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # Granice przestrzeni cech (do generowania punktów losowych)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    H_values = []
    for rep in range(n_repeats):
        # u_i — odległości z obserwacji do najbliższego sąsiada w danych
        sample_idx = rng.choice(n, n_samples, replace=False)
        Xs = X[sample_idx]
        dist_data, _ = nbrs.kneighbors(Xs)
        u = dist_data[:, 1]  # [:, 0] to byłaby odległość do siebie

        # w_i — odległości z losowych punktów przestrzeni do najbliższej obserwacji
        Y = rng.uniform(X_min, X_max, size=(n_samples, d))
        dist_rand, _ = nbrs.kneighbors(Y)
        w = dist_rand[:, 0]

        H = float(w.sum() / (u.sum() + w.sum()))
        H_values.append(H)

    return {
        "H_mean": float(np.mean(H_values)),
        "H_std":  float(np.std(H_values)),
        "n":      int(n),
        "d":      int(d),
        "n_samples": n_samples,
        "n_repeats": n_repeats,
        "all_values": H_values,
    }


def distance_coefficient_of_variation(X: np.ndarray, n_sample: int = 300,
                                       random_state: int = 42) -> dict:
    """
    Wyznacza współczynnik zmienności (CoV) odległości parami w danych.
    Niski CoV w wysokowymiarowych przestrzeniach to symptom curse of dimensionality.
    """
    rng = np.random.default_rng(random_state)
    if X.shape[0] > n_sample:
        idx = rng.choice(X.shape[0], n_sample, replace=False)
        X = X[idx]
    D = euclidean_distances(X)
    triu = D[np.triu_indices(X.shape[0], k=1)]
    return {
        "mean":  float(triu.mean()),
        "std":   float(triu.std()),
        "CoV":   float(triu.std() / triu.mean()),
        "min":   float(triu.min()),
        "max":   float(triu.max()),
    }


def main():
    print("=" * 70)
    print(" 11 — Statystyka Hopkinsa (tendencja klastrowa danych)")
    print("=" * 70)

    print("\n[1/4] Wczytywanie danych ...")
    customers = pd.read_pickle(config.PATH_CUSTOMERS)
    X_full    = pd.read_pickle(config.PATH_CUSTOMERS_SCALED).values
    X_norfm   = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM).values

    # Przestrzeń 3D RFM — standaryzowana
    X_rfm = StandardScaler().fit_transform(
        customers[["Recency", "Frequency", "Monetary"]].values)

    spaces = {
        "3D_RFM":   (X_rfm,   "3D — tylko cechy RFM"),
        "32D_full": (X_full,  "32D — pełen zbiór cech"),
        "26D_noRFM":(X_norfm, "26D — bez cech RFM"),
    }

    print(f"      Klienci: {customers.shape[0]:,}")
    for name, (X, desc) in spaces.items():
        print(f"      {name:12s} ({X.shape[1]:2d} cech): {desc}")

    # ----- Statystyka Hopkinsa -----
    print(f"\n[2/4] Wyznaczanie statystyki Hopkinsa "
          f"(n_samples=500, n_repeats=10, łącznie 5 000 punktów próbnych) ...")
    print("      Interpretacja:")
    print("        H ≈ 0,50 → brak struktury klastrowej (dane losowe)")
    print("        H > 0,75 → silna tendencja klastrowa")
    print("        H ≈ 1,00 → bardzo wyraźna struktura\n")

    hopkins_rows = []
    for name, (X, desc) in spaces.items():
        result = hopkins_statistic(X, n_samples=500, n_repeats=10,
                                    random_state=config.RANDOM_STATE)
        interpretation = ("brak struktury" if result["H_mean"] < 0.55
                          else "słaba" if result["H_mean"] < 0.65
                          else "umiarkowana" if result["H_mean"] < 0.75
                          else "silna" if result["H_mean"] < 0.90
                          else "bardzo silna")
        print(f"      {name:12s}  H = {result['H_mean']:.4f} ± {result['H_std']:.4f}"
              f"   ({interpretation})")
        hopkins_rows.append({
            "space": name,
            "description": desc,
            "n_obs": result["n"],
            "n_features": result["d"],
            "n_samples": result["n_samples"],
            "n_repeats": result["n_repeats"],
            "H_mean": round(result["H_mean"], 4),
            "H_std":  round(result["H_std"], 4),
            "interpretation": interpretation,
        })

    # ----- Współczynnik zmienności odległości -----
    print(f"\n[3/4] Współczynnik zmienności (CoV) odległości euklidesowych ...")
    print("      (niski CoV w wysokowymiarowej przestrzeni = symptom curse of dim)\n")
    cov_rows = []
    for name, (X, desc) in spaces.items():
        result = distance_coefficient_of_variation(X, n_sample=300,
                                                    random_state=config.RANDOM_STATE)
        diagnostic = ("OK" if result["CoV"] > 0.30
                      else "uwaga — curse of dim")
        print(f"      {name:12s}  CoV = {result['CoV']:.3f}   "
              f"(mean={result['mean']:.3f}, std={result['std']:.3f})   {diagnostic}")
        cov_rows.append({
            "space": name,
            "mean_distance": round(result["mean"], 3),
            "std_distance":  round(result["std"], 3),
            "CoV":           round(result["CoV"], 4),
            "diagnostic":    diagnostic,
        })

    # ----- Zapisz tabele -----
    print(f"\n[4/4] Zapis wyników ...")
    pd.DataFrame(hopkins_rows).to_csv(
        os.path.join(config.RESULTS_DIR, "hopkins_statistic.csv"), index=False)
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(config.RESULTS_DIR, "distance_stats.csv"), index=False)

    # ----- Wykres -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Hopkins
    ax = axes[0]
    space_names = [r["space"] for r in hopkins_rows]
    H_means     = [r["H_mean"] for r in hopkins_rows]
    H_stds      = [r["H_std"]  for r in hopkins_rows]
    colors = ["#2ecc71" if h >= 0.75 else "#f39c12" if h >= 0.55 else "#e74c3c"
              for h in H_means]
    bars = ax.bar(space_names, H_means, yerr=H_stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=8)
    for bar, h in zip(bars, H_means):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1,
                label="H = 0,50 (losowe)")
    ax.axhline(0.75, color="green", linestyle="--", linewidth=1.2,
                label="H = 0,75 (próg silnej struktury)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Statystyka Hopkinsa H")
    ax.set_title("Tendencja klastrowa danych w trzech przestrzeniach cech",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    # CoV odległości
    ax = axes[1]
    cov_values = [r["CoV"] for r in cov_rows]
    colors2 = ["#2ecc71" if c > 0.30 else "#e74c3c" for c in cov_values]
    bars2 = ax.bar(space_names, cov_values, color=colors2,
                    edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars2, cov_values):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.01,
                f"{c:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(0.30, color="red", linestyle="--", linewidth=1.2,
                label="próg curse of dim (CoV = 0,30)")
    ax.set_ylim(0, max(cov_values) * 1.25)
    ax.set_ylabel("CoV = std(d) / mean(d)")
    ax.set_title("Współczynnik zmienności odległości euklidesowych",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Diagnostyka jakości przestrzeni cech",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "fig_hopkins.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"      ✓ results/hopkins_statistic.csv")
    print(f"      ✓ results/distance_stats.csv")
    print(f"      ✓ results/fig_hopkins.png")
    print("\n[OK] Statystyka Hopkinsa wyznaczona.\n")


if __name__ == "__main__":
    main()
