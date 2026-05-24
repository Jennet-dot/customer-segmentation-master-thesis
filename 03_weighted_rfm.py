"""
03_weighted_rfm.py
------------------
Weighted RFM (Shih i Liu, 2005) — model BAZOWY #2.

Metoda (sekcja 2.2 pracy, wzór 10):
    weighted_RFM = w_R * R + w_F * F + w_M * M

Implementujemy DWA warianty doboru wag:
  (A) Literaturowy:  w_R = 0.2, w_F = 0.3, w_M = 0.5
      (monetary i frequency ważniejsze niż recency — typowa konwencja w branży e-commerce)
  (B) Data-driven:   wagi z entropii Shannona — zmienne o większej zmienności w danych
      otrzymują większą wagę. Patrz: Lang (2022) cytowane w pracy.

Score wRFM klasyfikujemy do 5 segmentów kwantylowo (segmenty: 1 = najsłabszy, 5 = najlepszy).

Uruchomienie:
    python 03_weighted_rfm.py

Wejście:  rfm_scores.pkl
Wyjście:  labels_wrfm.pkl              — etykiety wariantu literaturowego (główne)
          labels_wrfm_entropy.pkl      — etykiety wariantu entropy (porównawcze)
          wrfm_weights.pkl             — dict z obu zestawami wag
"""

import os
import pickle
import numpy as np
import pandas as pd

import config


def entropy_weights(scores: pd.DataFrame) -> dict:
    """
    Wyznacza wagi metodą entropii Shannona.
    Zmienna o większej zmienności (wyższej entropii znormalizowanej) → mniejsza waga d_j;
    klasycznie waga finalna = d_j / sum(d_j), gdzie d_j = 1 - E_j.
    Zwraca dict {'R': w_R, 'F': w_F, 'M': w_M}.
    """
    # Macierz score'ów [N x 3]
    X = scores[["R_score", "F_score", "M_score"]].astype(float).values
    # Normalizacja kolumnami (proporcje)
    P = X / X.sum(axis=0, keepdims=True)
    # Entropia każdej kolumny
    n = X.shape[0]
    k = 1.0 / np.log(n)
    # Bezpieczne log (log(0) → 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    E = -k * (P * log_P).sum(axis=0)
    # Stopień zróżnicowania
    d = 1.0 - E
    # Normalizacja wag
    w = d / d.sum()
    return {"R": float(w[0]), "F": float(w[1]), "M": float(w[2])}


def compute_wrfm_segments(scores: pd.DataFrame, weights: dict,
                          n_segments: int = config.WRFM_N_SEGMENTS) -> pd.Series:
    """
    Liczy ważony score wRFM i dzieli klientów na n_segments kwantyli.
    Etykiety: 1 = najsłabszy segment, n_segments = najlepszy.
    """
    wrfm = (weights["R"] * scores["R_score"]
            + weights["F"] * scores["F_score"]
            + weights["M"] * scores["M_score"])

    # Kwantylowy podział na n_segments grup. rank() rozdziela ewentualne duplikaty.
    segments = pd.qcut(wrfm.rank(method="first"),
                       q=n_segments,
                       labels=list(range(1, n_segments + 1))).astype(int)
    return segments


def label_segments_business(segments: pd.Series, n_segments: int = config.WRFM_N_SEGMENTS) -> pd.Series:
    """
    Mapuje numeryczne segmenty 1..n na czytelne etykiety biznesowe.
    Dla 5 segmentów: 1=Lost, 2=At Risk, 3=Potential Loyalist, 4=Loyal, 5=Champions.
    """
    if n_segments == 5:
        name_map = {1: "Lost", 2: "At Risk", 3: "Potential Loyalist",
                    4: "Loyal", 5: "Champions"}
    else:
        name_map = {i: f"Segment_{i}" for i in range(1, n_segments + 1)}
    return segments.map(name_map)


def main():
    print("=" * 70)
    print(" 03 — Weighted RFM (Shih i Liu, 2005) — model BAZOWY #2")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie score'ów RFM ...")
    scores = pd.read_pickle(os.path.join(config.DATA_DIR, "rfm_scores.pkl"))
    print(f"      Score'ów do przetworzenia: {len(scores):,}")

    # ----- Wariant A: wagi literaturowe -----
    print(f"\n[2/4] Wariant A — wagi literaturowe: {config.WRFM_WEIGHTS_LITERATURE}")
    seg_lit_int = compute_wrfm_segments(scores, config.WRFM_WEIGHTS_LITERATURE)
    seg_lit = label_segments_business(seg_lit_int)
    print("      Rozkład segmentów (literatura):")
    for s, n in seg_lit.value_counts().items():
        print(f"        {s:20s}  {n:>7,}  ({100 * n / len(seg_lit):5.2f}%)")

    # ----- Wariant B: wagi z entropii -----
    print(f"\n[3/4] Wariant B — wagi data-driven (entropia Shannona) ...")
    w_entropy = entropy_weights(scores)
    print(f"      Wyliczone wagi: R={w_entropy['R']:.4f}, "
          f"F={w_entropy['F']:.4f}, M={w_entropy['M']:.4f}")
    seg_ent_int = compute_wrfm_segments(scores, w_entropy)
    seg_ent = label_segments_business(seg_ent_int)
    print("      Rozkład segmentów (entropia):")
    for s, n in seg_ent.value_counts().items():
        print(f"        {s:20s}  {n:>7,}  ({100 * n / len(seg_ent):5.2f}%)")

    # ----- Zapis -----
    print("\n[4/4] Zapis etykiet ...")
    seg_lit.rename("wRFM").to_pickle(config.PATH_LABELS["wRFM"])
    seg_ent.rename("wRFM_entropy").to_pickle(
        os.path.join(config.DATA_DIR, "labels_wrfm_entropy.pkl"))
    with open(os.path.join(config.DATA_DIR, "wrfm_weights.pkl"), "wb") as f:
        pickle.dump({"literature": config.WRFM_WEIGHTS_LITERATURE,
                     "entropy": w_entropy}, f)

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['wRFM'])}        (wariant główny — literatura)")
    print(f"      ✓ labels_wrfm_entropy.pkl (wariant porównawczy — entropia)")
    print(f"      ✓ wrfm_weights.pkl        (obie wagi)")
    print("\n[OK] Weighted RFM zakończony.\n")


if __name__ == "__main__":
    main()
