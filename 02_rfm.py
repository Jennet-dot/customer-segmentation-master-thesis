"""
02_rfm.py
---------
Klasyczna segmentacja RFM (Hughes, 1994) — model BAZOWY #1.

Metoda (sekcja 2.2 pracy):
  1) Dla każdego klienta wyznacz Recency, Frequency, Monetary (gotowe z preprocessingu).
  2) Każdą zmienną podziel na 4 kwantyle i przypisz score 1-4 zgodnie z porządkiem:
       - Recency: niska = lepiej → kwantyl najmniejszych wartości dostaje 4
       - Frequency: wysoka = lepiej → kwantyl największych wartości dostaje 4
       - Monetary:  wysoka = lepiej → kwantyl największych wartości dostaje 4
  3) Klient otrzymuje 3-cyfrowy kod, np. "4-4-4" (najlepszy), "1-1-1" (najsłabszy).
  4) Wszystkich możliwych kombinacji jest 4^3 = 64 segmenty.
  5) Dodatkowo agregujemy do 5 interpretowalnych grup biznesowych (Champions, Loyal,
     Potential Loyalist, At Risk, Lost) na podstawie sumy R+F+M — ułatwia porównanie
     z modelami niebazowymi, które generują kilka klastrów.

Uruchomienie:
    python 02_rfm.py

Wejście:  customers.pkl
Wyjście:  labels_rfm.pkl — Series: Customer_ID -> etykieta segmentu
          rfm_scores.pkl — DataFrame ze score'ami R, F, M dla wRFM
"""

import os
import numpy as np
import pandas as pd

import config


def assign_rfm_scores(customers: pd.DataFrame, n_quantiles: int = 4) -> pd.DataFrame:
    """Przypisuje score'y 1..n_quantiles dla zmiennych R, F, M wg kwantyli."""
    scores = pd.DataFrame(index=customers.index)

    # Recency — odwrócona skala: mniejsza wartość = lepszy klient = wyższy score
    # qcut z labels od n..1 nadaje 4 wartości "najmłodszym" zakupom
    scores["R_score"] = pd.qcut(customers["Recency"], q=n_quantiles,
                                labels=list(range(n_quantiles, 0, -1)),
                                duplicates="drop").astype(int)

    # Frequency — naturalna skala: większa wartość = wyższy score.
    # rank("first") gwarantuje, że duplikaty (np. wszyscy z Frequency=1) zostaną rozdysponowane
    # po kwantylach proporcjonalnie zamiast wywołać błąd kwantyli.
    scores["F_score"] = pd.qcut(customers["Frequency"].rank(method="first"),
                                q=n_quantiles,
                                labels=list(range(1, n_quantiles + 1))).astype(int)

    # Monetary — większa wartość = wyższy score
    scores["M_score"] = pd.qcut(customers["Monetary"].rank(method="first"),
                                q=n_quantiles,
                                labels=list(range(1, n_quantiles + 1))).astype(int)

    return scores


def rfm_segment_code(scores: pd.DataFrame) -> pd.Series:
    """Tworzy 3-cyfrowy kod segmentu RFM, np. '444', '111', itp. — łącznie 4^3 = 64 segmenty."""
    return (scores["R_score"].astype(str) +
            scores["F_score"].astype(str) +
            scores["M_score"].astype(str))


def rfm_business_label(scores: pd.DataFrame) -> pd.Series:
    """
    Agregacja 64 segmentów RFM do 5 interpretowalnych grup biznesowych.
    Logika: suma R + F + M (zakres 3..12) + warunki dot. odpływu (niska Recency).

    Mapowanie zgodne z konwencją z pracy (str. 2.2):
        Champions          — wysokie R, F, M (RFM_sum >= 11)
        Loyal              — wysokie F, M, średnie R (RFM_sum 9-10)
        Potential Loyalist — wysokie R, niskie F (nowi, obiecujący, RFM_sum 7-8)
        At Risk            — niskie R, ale historycznie wysokie F/M (zagrożeni odejściem)
        Lost               — niskie wszystkie składowe (RFM_sum <= 5)
    """
    s = scores[["R_score", "F_score", "M_score"]].sum(axis=1)
    R, F, M = scores["R_score"], scores["F_score"], scores["M_score"]

    label = pd.Series("Potential Loyalist", index=scores.index)
    label[s >= 11] = "Champions"
    label[(s >= 9) & (s < 11) & (F >= 3)] = "Loyal"
    label[(R <= 2) & (F >= 3) & (M >= 3)] = "At Risk"
    label[s <= 5] = "Lost"
    return label


def main():
    print("=" * 70)
    print(" 02 — RFM (Hughes, 1994) — model BAZOWY #1")
    print("=" * 70)

    print(f"\n[1/4] Wczytywanie danych z {config.PATH_CUSTOMERS} ...")
    customers = pd.read_pickle(config.PATH_CUSTOMERS)
    print(f"      Klienci: {len(customers):,}")

    print(f"\n[2/4] Wyznaczanie score'ów R, F, M (kwantyle q={config.RFM_QUANTILES}) ...")
    scores = assign_rfm_scores(customers, n_quantiles=config.RFM_QUANTILES)
    scores["RFM_code"] = rfm_segment_code(scores)
    scores["RFM_sum"] = scores[["R_score", "F_score", "M_score"]].sum(axis=1)

    print(f"      Liczba unikalnych kodów RFM: {scores['RFM_code'].nunique()} "
          f"(max {config.RFM_QUANTILES ** 3})")

    print("\n[3/4] Agregacja 64 segmentów do 5 grup biznesowych ...")
    scores["RFM_label"] = rfm_business_label(scores)

    print("      Rozkład segmentów biznesowych:")
    distribution = scores["RFM_label"].value_counts()
    for seg, n in distribution.items():
        print(f"        {seg:20s}  {n:>7,}  ({100 * n / len(scores):5.2f}%)")

    print("\n[4/4] Zapis etykiet i score'ów ...")
    labels = scores["RFM_label"].rename("RFM")
    labels.to_pickle(config.PATH_LABELS["RFM"])
    scores.to_pickle(os.path.join(config.DATA_DIR, "rfm_scores.pkl"))

    print(f"      ✓ {os.path.basename(config.PATH_LABELS['RFM'])}")
    print(f"      ✓ rfm_scores.pkl  (R, F, M score'y do wykorzystania przez wRFM)")
    print("\n[OK] RFM zakończony.\n")


if __name__ == "__main__":
    main()
