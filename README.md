# Kod źródłowy do pracy magisterskiej

Repozytorium zawiera kod źródłowy wykorzystany w części empirycznej pracy magisterskiej dotyczącej segmentacji klientów sklepu internetowego z zastosowaniem metod RFM, ważonego RFM oraz wybranych algorytmów klasteryzacji.

## Temat analizy

Celem analizy było sprawdzenie, czy zaawansowane modele klasteryzacji są w stanie wykryć struktury segmentowe klientów o wyższej jakości niż modele bazowe oparte na klasycznym podejściu RFM oraz jego ważonej modyfikacji.

W badaniu wykorzystano dane transakcyjne sklepu internetowego obejmujące informacje o klientach, zamówieniach, wartościach transakcji, kategoriach produktów, metodach płatności, sposobach dostawy oraz innych cechach opisujących zachowania zakupowe klientów.

## Zakres repozytorium

Repozytorium obejmuje skrypty służące do:

- przygotowania i wstępnego przetwarzania danych,
- wyznaczenia zmiennych RFM,
- budowy klasycznej segmentacji RFM,
- budowy ważonej segmentacji RFM,
- trenowania modeli klasteryzacji,
- porównania modeli z użyciem miar jakości klasteryzacji,
- przygotowania tabel i wykresów wykorzystanych w pracy.

## Struktura plików

```text
.
├── 01_preprocessing.py
├── 02_rfm.py
├── 03_weighted_rfm.py
├── 04_kmeans.py
├── 04b_kmeans_norfm.py
├── 05_hierarchical.py
├── 05b_hierarchical_norfm.py
├── 06_birch_hdbscan.py
├── 06b_birch_hdbscan_norfm.py
├── 07_gmm.py
├── 07b_gmm_norfm.py
├── 08_metrics.py
├── 09_comparison.py
├── 10_per_model_plots.py
├── 11_hopkins_statistic.py
├── config.py
├── run_all.py
└── README.md
```

## Opis najważniejszych plików

| Plik | Opis |
|---|---|
| `01_preprocessing.py` | Przygotowanie danych do analizy, w tym czyszczenie, agregacja oraz utworzenie zbioru na poziomie klienta. |
| `02_rfm.py` | Budowa klasycznego modelu RFM na podstawie zmiennych Recency, Frequency i Monetary. |
| `03_weighted_rfm.py` | Budowa ważonego modelu RFM, w którym poszczególnym składowym przypisano określone wagi. |
| `04_kmeans.py` | Klasteryzacja metodą K-Means z wykorzystaniem zmiennych RFM oraz dodatkowych cech klienta. |
| `04b_kmeans_norfm.py` | Wariant ablacyjny K-Means bez zmiennych związanych z RFM. |
| `05_hierarchical.py` | Klasteryzacja hierarchiczna z wykorzystaniem zmiennych RFM oraz dodatkowych cech klienta. |
| `05b_hierarchical_norfm.py` | Wariant ablacyjny klasteryzacji hierarchicznej bez zmiennych związanych z RFM. |
| `06_birch_hdbscan.py` | Zastosowanie algorytmów BIRCH i HDBSCAN do segmentacji klientów. |
| `06b_birch_hdbscan_norfm.py` | Wariant ablacyjny modeli BIRCH i HDBSCAN bez zmiennych RFM. |
| `07_gmm.py` | Segmentacja z wykorzystaniem Gaussian Mixture Model. |
| `07b_gmm_norfm.py` | Wariant ablacyjny GMM bez zmiennych związanych z RFM. |
| `08_metrics.py` | Obliczenie miar jakości klasteryzacji, takich jak silhouette score, Calinski-Harabasz, Davies-Bouldin oraz inne wskaźniki. |
| `09_comparison.py` | Porównanie wyników uzyskanych przez poszczególne modele. |
| `10_per_model_plots.py` | Przygotowanie wykresów i wizualizacji dla poszczególnych modeli. |
| `11_hopkins_statistic.py` | Obliczenie statystyki Hopkinsa w celu oceny skłonności danych do tworzenia klastrów. |
| `config.py` | Plik konfiguracyjny zawierający ścieżki i podstawowe ustawienia projektu. |
| `run_all.py` | Skrypt umożliwiający uruchomienie pełnego procesu analitycznego. |

## Dane

Dane wykorzystane w analizie dotyczą transakcji klientów sklepu internetowego. Zbiór obejmuje m.in. informacje o:

- identyfikatorach klientów,
- datach zakupów,
- liczbie transakcji,
- wartości zakupów,
- kategoriach produktów,
- metodach płatności,
- sposobach dostawy,
- statusach zamówień,
- cechach demograficznych i geograficznych klientów.

Dane wejściowe nie zostały załączone do repozytorium, jeżeli ich udostępnienie jest ograniczone lub nie jest wymagane przez uczelnię. W takim przypadku repozytorium zawiera jedynie kod źródłowy wykorzystany do przeprowadzenia analizy.

## Wykorzystane metody

W pracy zastosowano następujące podejścia:

1. Klasyczna segmentacja RFM.
2. Ważona segmentacja RFM.
3. K-Means.
4. Klasteryzacja hierarchiczna.
5. BIRCH.
6. HDBSCAN.
7. Gaussian Mixture Model.
8. Warianty ablacyjne modeli bez zmiennych RFM.

## Ocena jakości klasteryzacji

Do oceny jakości segmentacji wykorzystano miary wewnętrzne, m.in.:

- silhouette score,
- Calinski-Harabasz index,
- Davies-Bouldin index,
- Within-Cluster Sum of Squares,
- Between-Cluster Sum of Squares,
- Dunn index,
- statystykę Hopkinsa.

Dodatkowo wyniki modeli zostały porównane z punktu widzenia interpretowalności oraz potencjalnej użyteczności biznesowej segmentów.

## Uruchomienie kodu

Aby uruchomić pełny proces analityczny, należy uruchomić plik:

```bash
python run_all.py
```

Poszczególne etapy analizy można również uruchamiać oddzielnie, zgodnie z numeracją plików.

## Wymagania

Kod został przygotowany w języku Python. Do uruchomienia analizy wymagane są standardowe biblioteki wykorzystywane w analizie danych i uczeniu maszynowym, m.in.:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
hdbscan
```

W przypadku problemów z biblioteką `hdbscan` może być konieczna jej osobna instalacja zgodnie z dokumentacją danej wersji Pythona i systemu operacyjnego.

## Uwagi techniczne

- Pliki wynikowe, wykresy oraz dane pośrednie mogą być zapisywane w osobnych folderach, zależnie od ustawień w pliku `config.py`.
- W repozytorium nie należy umieszczać plików zawierających dane osobowe, hasła, tokeny dostępu ani inne informacje poufne.
- W przypadku dużych plików danych zaleca się ich pominięcie w repozytorium i opisanie sposobu ich pozyskania lub przygotowania.

## Autor

Jennet Hamydova

## Informacja o pracy

Kod został przygotowany na potrzeby pracy magisterskiej dotyczącej porównania klasycznych i zaawansowanych metod segmentacji klientów w handlu elektronicznym.
