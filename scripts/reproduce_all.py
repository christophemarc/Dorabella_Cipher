#!/usr/bin/env python3
"""
reproduce_all.py
================
Master runner that reproduces all statistical analyses, Monte Carlo calibrations,
exact permutation tests, generative simulations, LOO-CV model selection, and all
four publication figures in one deterministic pass.
"""

import json
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

STEPS = [
    "01_geometry_and_wv_analysis.py",
    "02_elgar_corpus_compare.py",
    "03_generative_and_loocv.py",
    "06_peer_review_validations.py",
    "04_generate_figures.py",
    "04b_generate_figures_en.py",
    "05_build_html_report.py",
]


def main():
    t0 = time.time()
    print("=" * 80)
    print("DORABELLA CIPHER 2D GEOMETRIC & FORENSIC STUDY — FULL REPRODUCTION PIPELINE")
    print("=" * 80)
    for step in STEPS:
        path = os.path.join(SCRIPTS_DIR, step)
        print(f"\n>>> Running {step} ...")
        st = time.time()
        subprocess.run([sys.executable, path], check=True)
        print(f"    Completed in {time.time() - st:.2f}s")

    # Merge all 3 JSON outputs into a single results_summary.json
    summary = {}
    for key, fname in [
        ("geometry_and_wv", "results_01_geometry.json"),
        ("elgar_corpus", "results_02_corpus.json"),
        ("generative_and_loocv", "results_03_generative.json"),
        ("peer_review_validations", "results_06_validations.json"),
    ]:
        with open(os.path.join(DATA_DIR, fname)) as f:
            summary[key] = json.load(f)

    out_summary = os.path.join(DATA_DIR, "results_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"ALL PIPELINE STEPS SUCCEEDED in {time.time() - t0:.2f}s")
    print(f"Unified machine-readable summary: {out_summary}")
    print("=" * 80)


if __name__ == "__main__":
    main()
