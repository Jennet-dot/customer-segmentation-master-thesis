"""
08_metrics.py
-------------
Wszystkie mierniki z pracy magisterskiej — biznesowe + wewnętrzne + zewnętrzne.

MIERNIKI BIZNESOWE (Rozdział 1):
  - SRS  — Segment Revenue Share (udział segmentu w przychodach)
  - AOV  — Average Order Value (średnia wartość zamówienia)
  - CRR  — Customer Retention Rate (wskaźnik retencji)
  - CR   — Churn Rate (wskaźnik odpływu)
  - RaR  — Revenue at Risk (przychód zagrożony)

MIERNIKI WEWNĘTRZNE (sekcja 2.6, klasteryzacja bez ground-truth):
  - Silhouette (Rousseeuw, 1987)
  - Calinski-Harabasz index (Caliński i Harabasz, 1974)
  - Davies-Bouldin index (Dunn, 1974 wg pracy)
  - WSS / BSS (suma kwadratów wewnątrz/między klastrami)
  - Dunn index

MIERNIKI ZEWNĘTRZNE (sekcja 2.6, vs Customer_Segment jako ground-truth):
  - ARI (Hubert i Arabie, 1985)
  - Purity (Manning i in., 2008)
  - Entropy (Manning i in., 2008)
  - Variation of Information (Meila, 2007)
  - Pairwise ARI między modelami (model vs model)

Uruchomienie (po wszystkich poprzednich modułach):
    python 08_metrics.py

Wejście: customers.pkl, customers_scaled.pkl, ground_truth.pkl, labels_*.pkl
Wyjście (do katalogu results/):
    - business_metrics.csv         — mierniki biznesowe per (model, segment)
    - internal_metrics.csv         — mierniki wewnętrzne per model
    - external_metrics.csv         — mierniki zewnętrzne per model
    - pairwise_ari.csv             — macierz ARI między wszystkimi modelami
    - cluster_distribution.csv     — rozkład klientów w klastrach per model
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             mutual_info_score)

import config


# ============================================================================
# CZĘŚĆ 1: MIERNIKI BIZNESOWE
# ============================================================================

def segment_revenue_share(customers: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    SRS_k = sum(Monetary_i, i in k) / sum(Monetary_total) * 100%
    Zwraca udział każdego segmentu w przychodach (w %).
    """
    total_revenue = customers.loc[labels.index, "Monetary"].sum()
    per_segment = customers.loc[labels.index, "Monetary"].groupby(labels).sum()
    return (per_segment / total_revenue * 100).rename("SRS_pct")


def average_order_value(customers: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    AOV = Monetary / Frequency (na poziomie segmentu — suma wartości / suma transakcji).
    Zwraca średnią wartość zamówienia w segmencie.
    """
    grp = customers.loc[labels.index].groupby(labels)
    return (grp["Monetary"].sum() / grp["Frequency"].sum()).rename("AOV")


def retention_and_churn(transactions: pd.DataFrame, labels: pd.Series,
                         base_months: int, followup_months: int) -> pd.DataFrame:
    """
    Wyznacza CRR i CR dla każdego segmentu na podstawie surowych transakcji.

    Definicja (Kumar i Reinartz, 2018):
        CRR = (E - N) / S * 100%
        CR  = 100% - CRR
    gdzie:
        S — liczba klientów na początku okresu (aktywni w okresie bazowym)
        E — liczba klientów na końcu okresu (aktywni w okresie porównawczym)
        N — nowi klienci w okresie porównawczym (nie byli w bazowym)

    Operujemy na segmencie: klienci-baza ∩ segment vs klienci-followup ∩ segment.
    """
    transactions = transactions.copy()
    transactions["Date"] = pd.to_datetime(transactions["Date"])

    max_date = transactions["Date"].max()
    cutoff = max_date - pd.DateOffset(months=followup_months)

    base_tx     = transactions[transactions["Date"] <= cutoff]
    followup_tx = transactions[transactions["Date"]  > cutoff]

    base_customers     = set(base_tx["Customer_ID"].unique())
    followup_customers = set(followup_tx["Customer_ID"].unique())

    # Iterujemy po segmentach i liczymy CRR/CR
    rows = []
    for seg in sorted(labels.unique()):
        seg_customers = set(labels[labels == seg].index)
        S = len(seg_customers & base_customers)
        E = len(seg_customers & followup_customers)
        N = len((seg_customers & followup_customers) - base_customers)
        if S == 0:
            crr = np.nan
        else:
            crr = (E - N) / S * 100
        rows.append({"segment": seg, "CRR_pct": crr,
                     "CR_pct": (100 - crr) if not np.isnan(crr) else np.nan,
                     "S": S, "E": E, "N": N})
    df = pd.DataFrame(rows).set_index("segment")
    df.index.name = labels.name
    return df


def revenue_at_risk(customers: pd.DataFrame, labels: pd.Series,
                    top_pct: float, recency_threshold_days: int) -> float:
    """
    RaR (Wixom, Yen, Relich, 2013):
        RaR = sum(Monetary_i, i: top 20% wg Monetary AND Recency_i > 60 dni)
              / sum(Monetary_total) * 100%

    Zwraca pojedynczy wskaźnik dla całego zbioru (nie per segment),
    zgodnie z definicją z pracy. Można też policzyć per segment — zwracamy oba.
    """
    aligned = customers.loc[labels.index]
    total = aligned["Monetary"].sum()

    # Próg "top X%" wg Monetary
    monetary_threshold = aligned["Monetary"].quantile(1 - top_pct)
    at_risk_mask = (aligned["Monetary"] >= monetary_threshold) & \
                   (aligned["Recency"] > recency_threshold_days)

    rar_global = aligned.loc[at_risk_mask, "Monetary"].sum() / total * 100
    return float(rar_global)


def revenue_at_risk_per_segment(customers: pd.DataFrame, labels: pd.Series,
                                 top_pct: float, recency_threshold_days: int) -> pd.Series:
    """RaR dla każdego segmentu osobno (jaka część przychodów segmentu jest zagrożona)."""
    aligned = customers.loc[labels.index].copy()
    aligned["__seg"] = labels
    monetary_threshold = aligned["Monetary"].quantile(1 - top_pct)
    aligned["__at_risk"] = ((aligned["Monetary"] >= monetary_threshold) &
                            (aligned["Recency"] > recency_threshold_days))
    by_seg = aligned.groupby("__seg")
    rar = by_seg.apply(
        lambda g: g.loc[g["__at_risk"], "Monetary"].sum() / g["Monetary"].sum() * 100
                  if g["Monetary"].sum() > 0 else np.nan)
    rar.index.name = labels.name
    rar.name = "RaR_pct"
    return rar


# ============================================================================
# CZĘŚĆ 2: MIERNIKI WEWNĘTRZNE (jakość klasteryzacji)
# ============================================================================

def wss_bss(X: np.ndarray, labels: np.ndarray):
    """
    WSS = sum_j sum_{x in C_j} ||x - c_j||^2          (suma kwadratów wewnątrz klastrów)
    BSS = sum_j |C_j| * ||c_j - c_global||^2          (suma kwadratów między klastrami)
    """
    global_centroid = X.mean(axis=0)
    wss = 0.0
    bss = 0.0
    for c in np.unique(labels):
        mask = labels == c
        Xc = X[mask]
        if len(Xc) == 0:
            continue
        centroid = Xc.mean(axis=0)
        wss += float(((Xc - centroid) ** 2).sum())
        bss += float(len(Xc) * ((centroid - global_centroid) ** 2).sum())
    return wss, bss


def dunn_index(X: np.ndarray, labels: np.ndarray, max_n: int = 5000,
               random_state: int = 42) -> float:
    """
    Dunn = min_{k != l} delta(C_k, C_l) / max_k Delta(C_k)
        delta(C_k, C_l) — najmniejsza odległość między obiektami z różnych klastrów
        Delta(C_k)      — średnica klastra k (największa odległość wewnątrz klastra)

    Dla wydajności (O(n^2) odległości parami) ograniczamy do próbki max_n.
    """
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_n:
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X = X[idx]
        labels = labels[idx]

    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    # Średnice klastrów
    diameters = []
    cluster_points = {}
    for c in unique:
        Xc = X[labels == c]
        cluster_points[c] = Xc
        if len(Xc) < 2:
            diameters.append(0.0)
        else:
            d = pdist(Xc)
            diameters.append(float(d.max()) if len(d) > 0 else 0.0)
    max_diam = max(diameters)
    if max_diam == 0:
        return float("nan")

    # Najmniejsza odległość między parami klastrów
    min_inter = float("inf")
    for i, c1 in enumerate(unique):
        for c2 in unique[i + 1:]:
            d = cdist(cluster_points[c1], cluster_points[c2])
            min_d = float(d.min())
            if min_d < min_inter:
                min_inter = min_d

    return min_inter / max_diam


def internal_metrics(X: np.ndarray, labels: np.ndarray,
                      drop_noise: bool = True,
                      silhouette_sample: int = 10000,
                      random_state: int = 42) -> dict:
    """
    Liczy zestaw mierników wewnętrznych dla pojedynczego modelu.
    Jeśli drop_noise=True i występuje etykieta -1 (HDBSCAN), usuwa szum przed obliczeniami.
    """
    rng = np.random.default_rng(random_state)

    if drop_noise:
        mask = labels != -1
        X_eval = X[mask]
        lab_eval = labels[mask]
    else:
        X_eval = X
        lab_eval = labels

    unique = np.unique(lab_eval)
    n_clusters = len(unique)
    if n_clusters < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan, "WSS": np.nan, "BSS": np.nan,
                "dunn": np.nan, "n_clusters": int(n_clusters)}

    # Silhouette i Calinski-Harabasz na próbce dla wydajności
    if X_eval.shape[0] > silhouette_sample:
        sample_idx = rng.choice(X_eval.shape[0], size=silhouette_sample, replace=False)
        Xs, ls = X_eval[sample_idx], lab_eval[sample_idx]
    else:
        Xs, ls = X_eval, lab_eval

    sil = float(silhouette_score(Xs, ls))
    ch  = float(calinski_harabasz_score(Xs, ls))
    db  = float(davies_bouldin_score(X_eval, lab_eval))  # tańszy, można na całości
    wss, bss = wss_bss(X_eval, lab_eval)
    dunn = dunn_index(X_eval, lab_eval, max_n=5000, random_state=random_state)

    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db,
            "WSS": float(wss), "BSS": float(bss), "dunn": dunn,
            "n_clusters": int(n_clusters)}


# ============================================================================
# CZĘŚĆ 3: MIERNIKI ZEWNĘTRZNE (vs ground truth)
# ============================================================================

def contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Tabela kontyngencji n_ij = |X_i ∩ Y_j|."""
    df = pd.DataFrame({"t": labels_true, "p": labels_pred})
    return pd.crosstab(df["t"], df["p"]).values


def purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_j max_i n_ij     (Manning i in., 2008)
    Wartość 1 = pełna czystość, 0 = brak zgodności.
    """
    cm = contingency(labels_true, labels_pred)
    return float(cm.max(axis=0).sum() / cm.sum())


def cluster_entropy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Entropia klastrów (Manning i in., 2008):
        E = -sum_j p_j * sum_i (p_ij / p_j) * log(p_ij / p_j)
    Niska wartość = klastry jednorodne klasowo.
    """
    cm = contingency(labels_true, labels_pred).astype(float)
    N = cm.sum()
    if N == 0:
        return float("nan")
    p_j = cm.sum(axis=0) / N      # udział obiektów w klastrze j
    total = 0.0
    for j in range(cm.shape[1]):
        col = cm[:, j]
        if col.sum() == 0:
            continue
        proportions = col / col.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(proportions > 0, np.log(proportions), 0.0)
        ent_j = -(proportions * log_p).sum()
        total += p_j[j] * ent_j
    return float(total)


def variation_of_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    VI(X, Y) = H(X) + H(Y) - 2*I(X, Y)        (Meilă, 2007)
    Niska wartość = duża zgodność podziałów.
    """
    # H(X)
    _, counts_t = np.unique(labels_true, return_counts=True)
    p_t = counts_t / counts_t.sum()
    H_t = -(p_t * np.log(p_t)).sum()

    # H(Y)
    _, counts_p = np.unique(labels_pred, return_counts=True)
    p_p = counts_p / counts_p.sum()
    H_p = -(p_p * np.log(p_p)).sum()

    # I(X, Y) — mutual_info_score liczy z log naturalnym
    I = mutual_info_score(labels_true, labels_pred)

    return float(H_t + H_p - 2 * I)


def external_metrics(labels_true: pd.Series, labels_pred: pd.Series,
                      drop_noise: bool = True) -> dict:
    """Komplet mierników zewnętrznych vs ground truth."""
    # Wyrównanie indeksów + (opcjonalnie) usunięcie szumu HDBSCAN
    common = labels_true.index.intersection(labels_pred.index)
    yt = labels_true.loc[common].values
    yp = labels_pred.loc[common].values

    if drop_noise:
        mask = yp != -1
        yt = yt[mask]
        yp = yp[mask]

    if len(yt) == 0 or len(np.unique(yp)) < 1:
        return {"ARI": np.nan, "purity": np.nan,
                "entropy": np.nan, "VI": np.nan}

    return {
        "ARI":     float(adjusted_rand_score(yt, yp)),
        "purity":  purity(yt, yp),
        "entropy": cluster_entropy(yt, yp),
        "VI":      variation_of_information(yt, yp),
    }


def pairwise_ari(all_labels: dict, drop_noise: bool = True) -> pd.DataFrame:
    """Macierz ARI dla każdej pary modeli. Diagonal = 1.0 z definicji."""
    models = list(all_labels.keys())
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for i, m1 in enumerate(models):
        for m2 in models[i:]:
            l1 = all_labels[m1]
            l2 = all_labels[m2]
            common = l1.index.intersection(l2.index)
            y1, y2 = l1.loc[common].values, l2.loc[common].values
            if drop_noise:
                mask = (y1 != -1) & (y2 != -1)
                y1, y2 = y1[mask], y2[mask]
            ari = float(adjusted_rand_score(y1, y2))
            mat.loc[m1, m2] = ari
            mat.loc[m2, m1] = ari
    return mat


# ============================================================================
# CZĘŚĆ 4: GŁÓWNA PROCEDURA
# ============================================================================

def main():
    print("=" * 70)
    print(" 08 — Mierniki: biznesowe + wewnętrzne + zewnętrzne")
    print("=" * 70)

    # ---- Dane ----
    print("\n[1/5] Wczytywanie danych ...")
    customers       = pd.read_pickle(config.PATH_CUSTOMERS)
    X_scaled        = pd.read_pickle(config.PATH_CUSTOMERS_SCALED)
    X_scaled_norfm  = pd.read_pickle(config.PATH_CUSTOMERS_SCALED_NORFM)
    ground_truth    = pd.read_pickle(config.PATH_GROUND_TRUTH)
    transactions    = pd.read_csv(config.INPUT_CSV)
    transactions["Date"] = pd.to_datetime(transactions["Date"],
                                          format="%m/%d/%Y", errors="coerce")
    transactions = transactions.dropna(subset=["Customer_ID", "Date"]).copy()
    transactions["Customer_ID"] = transactions["Customer_ID"].astype(np.int64)

    all_labels = {}
    for name in config.MODELS_ORDER:
        path = config.PATH_LABELS[name]
        if os.path.exists(path):
            all_labels[name] = pd.read_pickle(path)
            print(f"      Wczytano {name:>14s}: {all_labels[name].nunique()} unikalnych etykiet")
        else:
            print(f"      [POMINIĘTO] {name}: brak pliku {os.path.basename(path)}")

    # ---- Rozkład klastrów ----
    print("\n[2/5] Liczenie rozkładów klastrów ...")
    rows = []
    for name, lab in all_labels.items():
        for seg, n in lab.value_counts().sort_index().items():
            rows.append({"model": name, "segment": str(seg), "n_customers": int(n),
                         "share_pct": 100 * n / len(lab)})
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_DIR, "cluster_distribution.csv"), index=False)

    # ---- Mierniki BIZNESOWE per (model, segment) ----
    print("\n[3/5] Mierniki biznesowe (SRS, AOV, CRR, CR, RaR) ...")
    biz_rows = []
    rar_global = {}
    for name, lab in all_labels.items():
        print(f"      {name} ...")
        srs = segment_revenue_share(customers, lab)
        aov = average_order_value(customers, lab)
        rc  = retention_and_churn(transactions, lab,
                                  config.CRR_BASE_MONTHS, config.CRR_FOLLOWUP_MONTHS)
        rar_seg = revenue_at_risk_per_segment(customers, lab,
                                              config.RAR_TOP_PCT,
                                              config.RAR_RECENCY_DAYS)
        rar_global[name] = revenue_at_risk(customers, lab,
                                            config.RAR_TOP_PCT,
                                            config.RAR_RECENCY_DAYS)
        seg_table = pd.DataFrame({"SRS_pct": srs, "AOV": aov, "RaR_pct": rar_seg}).join(rc)
        seg_table["model"] = name
        seg_table["segment"] = seg_table.index.astype(str)
        biz_rows.append(seg_table.reset_index(drop=True))
    biz_df = pd.concat(biz_rows, ignore_index=True)
    cols = ["model", "segment", "SRS_pct", "AOV", "CRR_pct", "CR_pct", "RaR_pct",
            "S", "E", "N"]
    biz_df = biz_df[cols]
    biz_df.to_csv(os.path.join(config.RESULTS_DIR, "business_metrics.csv"), index=False)

    rar_global_df = pd.DataFrame(
        [{"model": k, "RaR_global_pct": v} for k, v in rar_global.items()])
    rar_global_df.to_csv(os.path.join(config.RESULTS_DIR, "rar_global.csv"), index=False)

    # ---- Mierniki WEWNĘTRZNE per model ----
    # -----------------------------------------------------------------------
    # DWIE PRZESTRZENIE EWALUACJI:
    #
    # (A) WŁASNA przestrzeń każdego modelu — metodologicznie poprawna:
    #     RFM, wRFM  → 3D (Recency, Frequency, Monetary) standaryzowane
    #     _full      → 32D pełne
    #     _noRFM     → 26D bez RFM
    #     Uzasadnienie: silhouette w 32D dla RFM daje wartość 18x niższą niż
    #     w 3D (0.008 vs 0.155) z powodu curse of dimensionality (CoV=0.15 w 32D).
    #
    # (B) Wspólna przestrzeń PCA-15 (~77% wariancji) — porównywalna baza:
    #     Wszystkie modele w tej samej zredukowanej przestrzeni.
    #     RFM/wRFM → PCA-3 z ich 3D (max składowe); modele full → PCA-15 z 32D;
    #     modele noRFM → PCA-15 z 26D.
    #
    # Wyniki (A) trafiają do internal_metrics.csv (główne).
    # Wyniki (B) trafiają do internal_metrics_pca.csv (porównawcze).
    # -----------------------------------------------------------------------
    print("\n[4/5] Mierniki wewnętrzne (Silhouette, CH, DB, WSS, BSS, Dunn) ...")
    print("      [A] własna przestrzeń  [B] PCA-15 (wspólna baza)\n")

    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS

    rfm_cols_list = ["Recency", "Frequency", "Monetary"]
    X_rfm_3d = _SS().fit_transform(
        customers.loc[X_scaled.index, rfm_cols_list].values)

    # PCA dla każdego zestawu
    pca_full  = _PCA(n_components=15, random_state=config.RANDOM_STATE)
    pca_norfm = _PCA(n_components=15, random_state=config.RANDOM_STATE)
    pca_rfm3  = _PCA(n_components=3,  random_state=config.RANDOM_STATE)

    X_pca_full  = pca_full.fit_transform(X_scaled.values)
    X_pca_norfm = pca_norfm.fit_transform(X_scaled_norfm.values)
    X_pca_rfm3  = pca_rfm3.fit_transform(X_rfm_3d)

    print(f"      PCA wariancja: full={pca_full.explained_variance_ratio_.sum()*100:.1f}%, "
          f"noRFM={pca_norfm.explained_variance_ratio_.sum()*100:.1f}%, "
          f"RFM3D={pca_rfm3.explained_variance_ratio_.sum()*100:.1f}%\n")

    SPACE_OWN = {
        "RFM":                (X_rfm_3d,           "3D_rfm"),
        "wRFM":               (X_rfm_3d,           "3D_rfm"),
        "KMeans":             (X_scaled.values,    "32D_full"),
        "Hierarchical":       (X_scaled.values,    "32D_full"),
        "HDBSCAN":            (X_scaled.values,    "32D_full"),
        "GMM":                (X_scaled.values,    "32D_full"),
        "KMeans_noRFM":       (X_scaled_norfm.values, "26D_noRFM"),
        "Hierarchical_noRFM": (X_scaled_norfm.values, "26D_noRFM"),
        "HDBSCAN_noRFM":      (X_scaled_norfm.values, "26D_noRFM"),
        "GMM_noRFM":          (X_scaled_norfm.values, "26D_noRFM"),
    }
    SPACE_PCA = {
        "RFM":                (X_pca_rfm3,  "PCA3_rfm"),
        "wRFM":               (X_pca_rfm3,  "PCA3_rfm"),
        "KMeans":             (X_pca_full,  "PCA15_full"),
        "Hierarchical":       (X_pca_full,  "PCA15_full"),
        "HDBSCAN":            (X_pca_full,  "PCA15_full"),
        "GMM":                (X_pca_full,  "PCA15_full"),
        "KMeans_noRFM":       (X_pca_norfm, "PCA15_noRFM"),
        "Hierarchical_noRFM": (X_pca_norfm, "PCA15_noRFM"),
        "HDBSCAN_noRFM":      (X_pca_norfm, "PCA15_noRFM"),
        "GMM_noRFM":          (X_pca_norfm, "PCA15_noRFM"),
    }

    customer_idx = X_scaled.index
    int_rows_own, int_rows_pca = [], []

    for name, lab in all_labels.items():
        lab_aligned = lab.reindex(customer_idx)
        if pd.api.types.is_numeric_dtype(lab_aligned):
            lab_codes = lab_aligned.values.astype(int)
        else:
            _, lab_codes = np.unique(lab_aligned.values, return_inverse=True)

        is_hdbscan = name.startswith("HDBSCAN")
        X_own, sn_own = SPACE_OWN[name]
        X_pca_ev, sn_pca = SPACE_PCA[name]

        m_own = internal_metrics(X_own, lab_codes, drop_noise=is_hdbscan,
                                  random_state=config.RANDOM_STATE)
        m_own["model"] = name
        m_own["feature_space"] = sn_own
        int_rows_own.append(m_own)

        m_pca = internal_metrics(X_pca_ev, lab_codes, drop_noise=is_hdbscan,
                                  random_state=config.RANDOM_STATE)
        m_pca["model"] = name
        m_pca["feature_space"] = sn_pca
        int_rows_pca.append(m_pca)

        print(f"      {name:25s}  [A] sil={m_own['silhouette']:.4f} "
              f" [B] sil={m_pca['silhouette']:.4f}")

    cols_ord = ["feature_space", "n_clusters", "silhouette", "calinski_harabasz",
                "davies_bouldin", "WSS", "BSS", "dunn"]

    int_df_own = (pd.DataFrame(int_rows_own).set_index("model")[cols_ord]
                  .reindex([m for m in config.MODELS_ORDER
                             if m in pd.DataFrame(int_rows_own).set_index("model").index]))
    int_df_pca = (pd.DataFrame(int_rows_pca).set_index("model")[cols_ord]
                  .reindex([m for m in config.MODELS_ORDER
                             if m in pd.DataFrame(int_rows_pca).set_index("model").index]))

    int_df_own.to_csv(os.path.join(config.RESULTS_DIR, "internal_metrics.csv"))
    int_df_pca.to_csv(os.path.join(config.RESULTS_DIR, "internal_metrics_pca.csv"))

    # ---- Mierniki ZEWNĘTRZNE per model ----
    print("\n[5/5] Mierniki zewnętrzne (ARI, Purity, Entropy, VI) "
          "vs Customer_Segment ...")
    ext_rows = []
    for name, lab in all_labels.items():
        m = external_metrics(ground_truth, lab,
                              drop_noise=name.startswith("HDBSCAN"))
        m["model"] = name
        ext_rows.append(m)
        print(f"      {name:>14s}:  ARI={m['ARI']:.4f}  Purity={m['purity']:.4f}  "
              f"Entropy={m['entropy']:.4f}  VI={m['VI']:.4f}")
    ext_df = pd.DataFrame(ext_rows).set_index("model")[["ARI", "purity", "entropy", "VI"]]
    ext_df.to_csv(os.path.join(config.RESULTS_DIR, "external_metrics.csv"))

    # ---- Pairwise ARI między modelami ----
    print("\n      Pairwise ARI (model vs model) ...")
    pari = pairwise_ari(all_labels, drop_noise=True)
    pari.to_csv(os.path.join(config.RESULTS_DIR, "pairwise_ari.csv"))
    print(pari.round(3).to_string())

    print("\n[OK] Wszystkie mierniki policzone. Wyniki w katalogu results/.\n")


if __name__ == "__main__":
    main()
