"""
config.py
---------
Wspólne ścieżki i parametry dla wszystkich modeli.
Wszystkie pliki w projekcie korzystają z tych ustawień.
"""

import os

# ---------- Ścieżki ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wejściowy plik CSV. Zmień ścieżkę jeśli plik leży gdzie indziej.
INPUT_CSV = os.path.join(ROOT, "C:\\Users\\jhamy\\OneDrive\\Рабочий стол\\praca magisterska\\praca magisterska\\new_retail_data.csv")

# Pliki pośrednie (pickle) — przekazywane między modelami
PATH_CUSTOMERS = os.path.join(DATA_DIR, "customers.pkl")              # cechy klientów (RFM + pełne)
PATH_CUSTOMERS_SCALED = os.path.join(DATA_DIR, "customers_scaled.pkl") # standaryzowane (cechy pełne) do modeli z RFM
PATH_CUSTOMERS_SCALED_NORFM = os.path.join(DATA_DIR, "customers_scaled_norfm.pkl")  # standaryzowane bez RFM
PATH_FEATURE_COLS = os.path.join(DATA_DIR, "feature_columns.pkl")     # nazwy kolumn cech
PATH_GROUND_TRUTH = os.path.join(DATA_DIR, "ground_truth.pkl")        # Customer_Segment z danych

# Etykiety klastrów per model
PATH_LABELS = {
    "RFM": os.path.join(DATA_DIR, "labels_rfm.pkl"),
    "wRFM": os.path.join(DATA_DIR, "labels_wrfm.pkl"),
    "KMeans": os.path.join(DATA_DIR, "labels_kmeans.pkl"),
    "Hierarchical": os.path.join(DATA_DIR, "labels_hierarchical.pkl"),
    "HDBSCAN": os.path.join(DATA_DIR, "labels_hdbscan.pkl"),
    "GMM": os.path.join(DATA_DIR, "labels_gmm.pkl"),
}

# ---------- RFM ----------
# Klasyczny RFM wg Hughes (1994): podział każdej zmiennej na 4 kwantyle → 4^3 = 64 segmenty
RFM_QUANTILES = 4

# ---------- Weighted RFM ----------
# Domyślne wagi z literatury (Shih i Liu, 2005): monetary najważniejsze, recency najmniej
WRFM_WEIGHTS_LITERATURE = {"R": 0.2, "F": 0.3, "M": 0.5}
# Liczba segmentów wRFM (klasyfikacja zagregowanego score'a kwantylowo)
WRFM_N_SEGMENTS = 5

# ---------- Klasteryzacja (K-means, Hierarchical, GMM) ----------
# Zakres k do automatycznego doboru (silhouette + metoda łokcia)
K_RANGE = list(range(2, 11))
RANDOM_STATE = 42

# Hierarchical: dla 85k klientów pełne grupowanie jest niewykonalne pamięciowo (O(n^2)).
# Próbkujemy HIER_SAMPLE_SIZE klientów, dopasowujemy hierarchię i propagujemy etykiety
# do pozostałych klientów przez najbliższy centroid klastra (1-NN).
HIER_SAMPLE_SIZE = 10000
HIER_LINKAGE = "ward"  # metoda Warda – minimalizacja SSE przy łączeniu klastrów (Ward, 1963)

# ---------- BIRCH → HDBSCAN ----------
# Krok 1: BIRCH kompresuje N obserwacji do M sub-klastrów (CF-tree).
# Krok 2: HDBSCAN klastruje centroidy sub-klastrów (M << N).
# Krok 3: każdy klient dziedziczy etykietę swojego sub-klastra.
BIRCH_THRESHOLD = 4.0      # T — maksymalny promień sub-klastra (skalibrowany pod nasze dane:
                           # threshold=4.0 daje ~1800 sub-klastrów z 87k klientów = kompresja 48x)
BIRCH_BRANCHING = 50       # B — maksymalna liczba dzieci węzła wewnętrznego
BIRCH_N_CLUSTERS = None    # None → zwraca surowe sub-klastry CF-tree (bez globalnej klasteryzacji)

HDBSCAN_MIN_CLUSTER_SIZE = 20  # minimalna wielkość klastra (na poziomie sub-klastrów BIRCH)
HDBSCAN_MIN_SAMPLES = 5        # min_samples — gęstość lokalna

# ---------- Okna czasowe dla mierników biznesowych ----------
# Dane obejmują 2023-03 do 2024-02 (12 mies.) → bazowy 9 mies., porównawczy 3 mies.
CRR_BASE_MONTHS = 9
CRR_FOLLOWUP_MONTHS = 3

# Revenue at Risk: top 20% wg Monetary, którzy nie kupowali > 60 dni
RAR_TOP_PCT = 0.20
RAR_RECENCY_DAYS = 60

# Lista wszystkich modeli (kolejność w tabelach porównawczych)
MODELS_ORDER = ["RFM", "wRFM",
                "KMeans", "Hierarchical", "HDBSCAN", "GMM",
                "KMeans_noRFM", "Hierarchical_noRFM", "HDBSCAN_noRFM", "GMM_noRFM"]
BASELINE_MODELS = ["RFM", "wRFM"]
COMPARATIVE_MODELS_FULL = ["KMeans", "Hierarchical", "HDBSCAN", "GMM"]
COMPARATIVE_MODELS_NORFM = ["KMeans_noRFM", "Hierarchical_noRFM", "HDBSCAN_noRFM", "GMM_noRFM"]

# Aktualizacja słownika PATH_LABELS o warianty bez RFM
PATH_LABELS.update({
    "KMeans_noRFM":       os.path.join(DATA_DIR, "labels_kmeans_norfm.pkl"),
    "Hierarchical_noRFM": os.path.join(DATA_DIR, "labels_hierarchical_norfm.pkl"),
    "HDBSCAN_noRFM":      os.path.join(DATA_DIR, "labels_hdbscan_norfm.pkl"),
    "GMM_noRFM":          os.path.join(DATA_DIR, "labels_gmm_norfm.pkl"),
})

# Cechy NIE-RFM: do wykluczenia ze zbioru "no_rfm"
# (RFM bazowe + ich pochodne — wszystko co bezpośrednio zależy od R/F/M)
RFM_FEATURES_TO_EXCLUDE = ["Recency", "Frequency", "Monetary",
                            "AvgOrderValue", "StdOrderValue", "TotalItems"]
