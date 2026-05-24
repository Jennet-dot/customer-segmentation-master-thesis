"""
01_preprocessing.py
-------------------
Ładowanie surowych danych i przygotowanie zbioru do klasteryzacji.

Wykonywane kroki:
  1) Wczytanie CSV i czyszczenie (parsowanie dat, usunięcie braków w kluczowych polach).
  2) Agregacja transakcji do poziomu klienta (Customer_ID).
  3) Wyznaczenie cech RFM: Recency, Frequency, Monetary (potrzebnych dla RFM i wRFM).
  4) Wyznaczenie pełnego zbioru cech klienta (dla K-means, Hierarchical, HDBSCAN, GMM):
       - cechy numeryczne (RFM + demograficzne + behawioralne)
       - cechy kategoryczne kodowane one-hot / porządkowe
  5) Standaryzacja z-score wg wzorów (3)-(6) z pracy.
  6) Wyodrębnienie zmiennej Customer_Segment jako ground-truth dla mierników zewnętrznych.
  7) Zapisanie wyników do plików pickle (używane przez kolejne moduły 02-07).

Uruchomienie:
    python 01_preprocessing.py

Na wyjściu (w katalogu data/):
    - customers.pkl            — DataFrame z cechami klientów (nieprzeskalowany)
    - customers_scaled.pkl     — DataFrame ze standaryzowanymi cechami (do non-RFM modeli)
    - feature_columns.pkl      — dict: {'rfm': [...], 'full': [...]}
    - ground_truth.pkl         — Series: Customer_ID -> Customer_Segment
"""

import os
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------- Funkcje pomocnicze ----------
def safe_mode(series: pd.Series):
    """Najczęstsza wartość z Series; przy remisie pierwsza alfabetycznie. Toleruje NaN."""
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


# ---------- Główna procedura ----------
def main():
    print("=" * 70)
    print(" 01 — Preprocessing danych sprzedażowych")
    print("=" * 70)

    # ----- 1. Wczytanie CSV -----
    print(f"\n[1/7] Wczytywanie {config.INPUT_CSV} ...")
    df = pd.read_csv(config.INPUT_CSV)
    print(f"      Wczytano {len(df):,} transakcji, {df['Customer_ID'].nunique():,} klientów")

    # ----- 2. Czyszczenie -----
    print("\n[2/7] Czyszczenie danych ...")
    # Parsowanie daty (format M/D/Y)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")

    # Wymagamy poprawnego Customer_ID, daty i kwoty
    n0 = len(df)
    df = df.dropna(subset=["Customer_ID", "Date", "Total_Amount", "Amount"]).copy()
    df["Customer_ID"] = df["Customer_ID"].astype(np.int64)
    print(f"      Po usunięciu braków: {len(df):,} transakcji (usunięto {n0 - len(df):,})")

    # Data referencyjna do liczenia Recency = ostatnia data + 1 dzień
    ref_date = df["Date"].max() + pd.Timedelta(days=1)
    print(f"      Zakres dat: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"      Data referencyjna (Recency): {ref_date.date()}")

    # ----- 3. Agregacja do poziomu klienta + cechy RFM -----
    print("\n[3/7] Agregacja do poziomu klienta i wyznaczanie RFM ...")
    agg_basic = (
        df.groupby("Customer_ID")
          .agg(
              Recency=("Date",          lambda x: (ref_date - x.max()).days),
              Frequency=("Date",        "count"),                          # liczba transakcji
              Monetary=("Total_Amount", "sum"),                            # łączna wartość zakupów
              AvgOrderValue=("Total_Amount", "mean"),
              StdOrderValue=("Total_Amount", "std"),
              TotalItems=("Total_Purchases", "sum"),
              AvgRating=("Ratings",     "mean"),
              Age=("Age",               "mean"),
              UniqueCategories=("Product_Category", "nunique"),
              UniqueBrands=("Product_Brand", "nunique"),
              PendingOrReturned=("Order_Status",
                                 lambda x: (x.isin(["Pending", "Processing"])).sum()),
          )
    )
    agg_basic["StdOrderValue"] = agg_basic["StdOrderValue"].fillna(0.0)
    agg_basic["AvgRating"] = agg_basic["AvgRating"].fillna(agg_basic["AvgRating"].median())
    agg_basic["Age"] = agg_basic["Age"].fillna(agg_basic["Age"].median())

    # Pochodne wskaźniki behawioralne
    agg_basic["PendingRate"] = agg_basic["PendingOrReturned"] / agg_basic["Frequency"]
    print(f"      Liczba klientów po agregacji: {len(agg_basic):,}")

    # ----- 4. Cechy kategoryczne (dominanta per klient) -----
    print("\n[4/7] Cechy kategoryczne (dominanta per klient) ...")
    cat_cols_mode = ["Gender", "Income", "Country",
                     "Product_Category", "Payment_Method",
                     "Shipping_Method", "Feedback"]

    cat_agg = df.groupby("Customer_ID")[cat_cols_mode].agg(safe_mode)
    cat_agg.columns = [f"Dom_{c}" for c in cat_agg.columns]

    # Ground truth do mierników zewnętrznych — Customer_Segment per klient (dominanta)
    ground_truth = df.groupby("Customer_ID")["Customer_Segment"].agg(safe_mode)
    ground_truth.name = "Customer_Segment"

    # Łączymy
    customers = agg_basic.join(cat_agg).join(ground_truth)
    customers = customers.dropna(subset=["Dom_Gender", "Dom_Income", "Dom_Country",
                                         "Customer_Segment"])
    print(f"      Po usunięciu klientów z brakami kategorycznymi: {len(customers):,}")

    # ----- 5. Kodowanie cech -----
    print("\n[5/7] Kodowanie cech (porządkowe + one-hot) ...")
    # Income → porządkowe (Low=1, Medium=2, High=3)
    income_map = {"Low": 1, "Medium": 2, "High": 3}
    customers["Income_ord"] = customers["Dom_Income"].map(income_map)

    # Feedback → porządkowe (Bad=1, Average=2, Good=3, Excellent=4)
    feedback_map = {"Bad": 1, "Average": 2, "Good": 3, "Excellent": 4}
    customers["Feedback_ord"] = customers["Dom_Feedback"].map(feedback_map).fillna(2)

    # One-hot dla nominalnych
    onehot_cols = ["Dom_Gender", "Dom_Country", "Dom_Product_Category",
                   "Dom_Payment_Method", "Dom_Shipping_Method"]
    onehot = pd.get_dummies(customers[onehot_cols], prefix=onehot_cols, dtype=np.float32)
    customers = pd.concat([customers, onehot], axis=1)

    # ----- 6. Definicja zestawów cech -----
    # Zestaw "rfm" (do RFM i weighted RFM): tylko trzy klasyczne wskaźniki
    rfm_cols = ["Recency", "Frequency", "Monetary"]

    # Zestaw "full" (do K-means, Hierarchical, HDBSCAN, GMM): WSZYSTKIE dostępne cechy.
    # Wykluczamy: kolumny tekstowe (zastąpione one-hot/porządkowymi) oraz Customer_Segment
    # (ground truth — nie może być wśród cech klasteryzacji).
    drop_for_full = (["Dom_Income", "Dom_Feedback", "Customer_Segment", "PendingOrReturned"]
                     + onehot_cols)
    full_cols = [c for c in customers.columns if c not in drop_for_full]
    print(f"      Cechy RFM:  {rfm_cols}")
    print(f"      Cechy full: {len(full_cols)} kolumn (numeryczne + porządkowe + one-hot)")

    feature_columns = {"rfm": rfm_cols, "full": full_cols}

    # Zestaw "no_rfm" (do K-means_noRFM, Hierarchical_noRFM, HDBSCAN_noRFM, GMM_noRFM):
    # WSZYSTKIE cechy POZA RFM i pochodnymi RFM. Pozwala odpowiedzieć na pytanie:
    # czy zaawansowane modele potrafią odkryć sensowne segmenty bez bezpośredniego
    # dostępu do informacji R/F/M, korzystając wyłącznie z cech demograficzno-
    # behawioralnych?
    no_rfm_cols = [c for c in full_cols if c not in config.RFM_FEATURES_TO_EXCLUDE]
    print(f"      Cechy no_rfm: {len(no_rfm_cols)} kolumn "
          f"(pełen zbiór bez {len(full_cols) - len(no_rfm_cols)} cech RFM i pochodnych)")

    feature_columns = {"rfm": rfm_cols, "full": full_cols, "no_rfm": no_rfm_cols}

    # ----- 7. Standaryzacja (z-score, wzory 3-6 z pracy) -----
    print("\n[6/7] Standaryzacja zmiennych (z-score) ...")
    customers_full = customers[full_cols].astype(np.float64).copy()
    scaler_full = StandardScaler()
    customers_scaled = pd.DataFrame(
        scaler_full.fit_transform(customers_full),
        index=customers_full.index,
        columns=full_cols,
    )

    # Osobna standaryzacja dla zbioru no_rfm (cechy mają inne rozkłady niż w pełnym zbiorze,
    # więc standaryzacja musi być przeliczona; matematycznie subset standaryzowanego pełnego
    # zbioru NIE jest standardowo rozłożony, bo średnia/wariancja po wyborze podzbioru
    # cech może się zmienić niewiele, ale dobrą praktyką jest refit)
    customers_no_rfm = customers[no_rfm_cols].astype(np.float64).copy()
    scaler_no_rfm = StandardScaler()
    customers_scaled_no_rfm = pd.DataFrame(
        scaler_no_rfm.fit_transform(customers_no_rfm),
        index=customers_no_rfm.index,
        columns=no_rfm_cols,
    )

    # ----- 8. Zapis -----
    print("\n[7/7] Zapis artefaktów do katalogu data/ ...")
    customers.to_pickle(config.PATH_CUSTOMERS)
    customers_scaled.to_pickle(config.PATH_CUSTOMERS_SCALED)
    customers_scaled_no_rfm.to_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    with open(config.PATH_FEATURE_COLS, "wb") as f:
        pickle.dump(feature_columns, f)
    customers["Customer_Segment"].to_pickle(config.PATH_GROUND_TRUTH)

    print(f"      ✓ {os.path.basename(config.PATH_CUSTOMERS)}        ({len(customers):,} klientów, "
          f"{customers.shape[1]} kolumn)")
    print(f"      ✓ {os.path.basename(config.PATH_CUSTOMERS_SCALED)} "
          f"({len(customers_scaled):,} klientów, {customers_scaled.shape[1]} cech pełnych)")
    print(f"      ✓ {os.path.basename(config.PATH_CUSTOMERS_SCALED_NORFM)} "
          f"({len(customers_scaled_no_rfm):,} klientów, {customers_scaled_no_rfm.shape[1]} cech bez RFM)")
    print(f"      ✓ {os.path.basename(config.PATH_FEATURE_COLS)}")
    print(f"      ✓ {os.path.basename(config.PATH_GROUND_TRUTH)}")

    # Podsumowanie ground truth
    print("\n      Rozkład Customer_Segment (ground truth):")
    for seg, n in customers["Customer_Segment"].value_counts().items():
        print(f"        {seg:10s}  {n:>7,}  ({100 * n / len(customers):5.2f}%)")

    print("\n[OK] Preprocessing zakończony.\n")


if __name__ == "__main__":
    main()
