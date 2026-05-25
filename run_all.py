"""
run_all.py
----------
Uruchamia całą procedurę po kolei: preprocessing → wszystkie modele → mierniki → porównanie.

Uruchomienie:
    python run_all.py

Każdy krok jest uruchamiany jako oddzielny proces (subprocess) — taka sama logika
jak ręczne uruchamianie pliku po pliku. Pomaga to też uniknąć efektów ubocznych
(każdy moduł startuje z czystym importem).
"""

import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("01_preprocessing.py",        "Preprocessing"),
    ("02_rfm.py",                  "RFM (baseline)"),
    ("03_weighted_rfm.py",         "Weighted RFM (baseline)"),
    ("04_kmeans.py",               "K-means (cechy pełne)"),
    ("04b_kmeans_norfm.py",        "K-means (cechy bez RFM)"),
    ("05_hierarchical.py",         "Hierarchical Ward (cechy pełne)"),
    ("05b_hierarchical_norfm.py",  "Hierarchical Ward (cechy bez RFM)"),
    ("06_birch_hdbscan.py",        "BIRCH → HDBSCAN (cechy pełne)"),
    ("06b_birch_hdbscan_norfm.py", "BIRCH → HDBSCAN (cechy bez RFM)"),
    ("07_gmm.py",                  "Gaussian Mixture Models (cechy pełne)"),
    ("07b_gmm_norfm.py",           "Gaussian Mixture Models (cechy bez RFM)"),
    ("08_metrics.py",              "Wszystkie mierniki"),
    ("09_comparison.py",           "Tabela porównawcza + wykresy"),
    ("10_per_model_plots.py",      "Wykresy analityczne per model (50 PNG)"),
    ("11_hopkins_statistic.py",    "Statystyka Hopkinsa + diagnostyka CoV"),
    ("12_dendrogram.py",    "Dendrogramy klasteryzacji hierarchicznej Warda"),
]


def main():
    print("\n" + "█" * 70)
    print("  URUCHAMIANIE PEŁNEGO PIPELINE'U SEGMENTACJI KLIENTÓW")
    print("█" * 70)

    overall_start = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        print(f"\n>>> [{i}/{len(STEPS)}] {desc}  ({script})")
        start = time.time()
        result = subprocess.run([sys.executable, os.path.join(ROOT, script)],
                                 cwd=ROOT)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n[BŁĄD] {script} zakończony z kodem {result.returncode}.")
            print("Pipeline przerwany.")
            sys.exit(result.returncode)
        print(f"<<< [{i}/{len(STEPS)}] {desc} OK  ({elapsed:.1f}s)")

    total = time.time() - overall_start
    print("\n" + "█" * 70)
    print(f"  PIPELINE ZAKOŃCZONY. Łączny czas: {total:.1f}s ({total/60:.1f} min)")
    print("█" * 70)
    print(f"\nWyniki w katalogu: {os.path.join(ROOT, 'results')}")
    print("Etykiety klastrów: data/labels_*.pkl")
    print("Tabela zbiorcza:   results/summary_table.csv")
    print("Wykresy:           results/fig_*.png\n")


if __name__ == "__main__":
    main()
