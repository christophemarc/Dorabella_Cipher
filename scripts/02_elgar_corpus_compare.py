#!/usr/bin/env python3
"""
02_elgar_corpus_compare.py
==========================
Cross-document forensic comparison of Edward Elgar's complete surviving
semicircle-glyph corpus:
  - Liszt Fragment (10 April 1886, N=18, std & sensitivity alt reading)
  - Dorabella Cipher (14 July 1897, N=87, Hauer 2025 consensus)
  - Marco Elgar Notebook (post-May 1924, N=39 & N=47, exact key)
  - 500 authentic English controls encrypted with Elgar's exact 1924 Marco key

Outputs:
  ../data/results_02_corpus.json
"""

import json
import os
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, fisher_exact, chi2_contingency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ORI_LIST = list("ABCDEFGH")
ORI_MAP = {c: i for i, c in enumerate(ORI_LIST)}
W_SET = {1, 2, 5, 6}

# Same authentic English literary text for Marco-key control encryption
from importlib.machinery import SourceFileLoader
_m01 = SourceFileLoader("m01", os.path.join(os.path.dirname(__file__), "01_geometry_and_wv_analysis.py")).load_module()
ENGLISH_SAMPLE = _m01.ENGLISH_SAMPLE
exact_wv_zscore = _m01.exact_wv_zscore
compute_ic = _m01.compute_ic


def max_no_rep_arc(arcs):
    best, cur = 1, 1
    for i in range(1, len(arcs)):
        if arcs[i] != arcs[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def doc_profile(name, toks, rng, n_mc=20000):
    n = len(toks)
    oris = np.array([ORI_MAP[t[0]] for t in toks])
    arcs = np.array([int(t[1]) for t in toks])
    is_W = np.array([o in W_SET for o in oris])
    ori_counts = [int(np.sum(oris == i)) for i in range(8)]
    ori_freq = [c / n for c in ori_counts]
    arc_counts = [int(np.sum(arcs == a)) for a in (1, 2, 3)]

    cross, exp_c, sd_c, z_wv = exact_wv_zscore(oris, W_SET)
    t180 = ((oris[:-1] - oris[1:]) % 8 == 4)
    tsam = (arcs[:-1] == arcs[1:])
    same_axis = ((oris[:-1] - oris[1:]) % 4 == 0)

    c180 = int(t180.sum())
    c180_same = int((t180 & tsam).sum())
    c_axis = int(same_axis.sum())

    # Shuffle null for 180°
    mc_180 = np.zeros(n_mc)
    mc_180s = np.zeros(n_mc)
    for k in range(n_mc):
        p = rng.permutation(n)
        po, pa = oris[p], arcs[p]
        tt = ((po[:-1] - po[1:]) % 8 == 4)
        mc_180[k] = tt.sum()
        mc_180s[k] = (tt & (pa[:-1] == pa[1:])).sum()

    z180 = float((c180 - mc_180.mean()) / (mc_180.std() + 1e-12))
    z180s = float((c180_same - mc_180s.mean()) / (mc_180s.std() + 1e-12))

    # F contacts with V
    f_to_v, f_tot = 0, 0
    for i, o in enumerate(oris):
        if o == 5:
            if i > 0:
                f_tot += 1
                f_to_v += int(oris[i - 1] not in W_SET)
            if i < n - 1:
                f_tot += 1
                f_to_v += int(oris[i + 1] not in W_SET)

    return {
        "name": name,
        "n": n,
        "unique_symbols": len(set(toks)),
        "ic": round(compute_ic(toks), 5),
        "ori_counts": {ORI_LIST[i]: ori_counts[i] for i in range(8)},
        "ori_pct": {ORI_LIST[i]: round(100.0 * ori_freq[i], 2) for i in range(8)},
        "arc_counts": arc_counts,
        "arc_pct": [round(100.0 * c / n, 1) for c in arc_counts],
        "n_W": int(is_W.sum()),
        "pct_W": round(100.0 * float(is_W.mean()), 2),
        "crossings_wv": cross,
        "z_wv": round(z_wv, 3),
        "flips_180_all": c180,
        "flips_180_exp": round(float(mc_180.mean()), 2),
        "z_180_all": round(z180, 3),
        "flips_180_same": c180_same,
        "flips_180_same_exp": round(float(mc_180s.mean()), 2),
        "z_180_same": round(z180s, 3),
        "same_axis_pct": round(100.0 * c_axis / (n - 1), 1),
        "max_no_rep_arc": max_no_rep_arc(arcs),
        "f_to_v": f"{f_to_v}/{f_tot}"
    }


def corr_and_perm(v1, v2, rng, n_perm=50000):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    r, p_two = pearsonr(v1, v2)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    # Exact permutation test over 8! = 40,320 (sampled 50k)
    ge = 0
    for _ in range(n_perm):
        rp = np.corrcoef(v1, rng.permutation(v2))[0, 1]
        if rp >= r - 1e-12:
            ge += 1
    return {
        "pearson_r": round(float(r), 4),
        "p_two_tailed": round(float(p_two), 5),
        "p_perm_one_tailed": round(ge / n_perm, 5),
        "cosine": round(cos, 4)
    }


def marco_key_controls(letter_to_tok, length, rng, n_ctrl=500):
    n_text = len(ENGLISH_SAMPLE)
    w_pcts, z_wvs, f180s, f180ss, mnr = [], [], [], [], []
    for _ in range(n_ctrl):
        st = int(rng.integers(0, n_text - length))
        sub = ENGLISH_SAMPLE[st:st + length]
        toks = [letter_to_tok[c] for c in sub]
        oris = np.array([ORI_MAP[t[0]] for t in toks])
        arcs = np.array([int(t[1]) for t in toks])
        w_pcts.append(100.0 * np.mean([o in W_SET for o in oris]))
        z_wvs.append(exact_wv_zscore(oris, W_SET)[3])
        t180 = ((oris[:-1] - oris[1:]) % 8 == 4)
        f180s.append(int(t180.sum()))
        f180ss.append(int((t180 & (arcs[:-1] == arcs[1:])).sum()))
        mnr.append(max_no_rep_arc(arcs))
    w_pcts, z_wvs, f180s, f180ss, mnr = map(np.array, (w_pcts, z_wvs, f180s, f180ss, mnr))
    return {
        "n_controls": n_ctrl,
        "length": length,
        "pct_W_mean": round(float(w_pcts.mean()), 2),
        "pct_W_sd": round(float(w_pcts.std()), 2),
        "z_wv_mean": round(float(z_wvs.mean()), 3),
        "z_wv_sd": round(float(z_wvs.std()), 3),
        "flips_180_mean": round(float(f180s.mean()), 2),
        "flips_180_sd": round(float(f180s.std()), 2),
        "flips_180_same_mean": round(float(f180ss.mean()), 2),
        "flips_180_same_sd": round(float(f180ss.std()), 2),
        "max_no_rep_mean": round(float(mnr.mean()), 2),
        "count_z_wv_le_dorabella": int(np.sum(z_wvs <= -4.756)),
        "count_180_ge_27": int(np.sum(f180s >= 27)),
        "count_180_same_ge_13": int(np.sum(f180ss >= 13)),
        "samples_for_plot": {
            "pct_W": [round(float(x), 2) for x in w_pcts[:250]],
            "z_wv": [round(float(x), 3) for x in z_wvs[:250]],
            "flips_180_pct": [round(100.0 * float(x) / (length - 1), 2) for x in f180s[:250]]
        }
    }


def main():
    rng = np.random.default_rng(20260410)
    with open(os.path.join(DATA_DIR, "elgar_corpus.json")) as f:
        cdata = json.load(f)

    docs = cdata["documents"]
    profiles = {}
    print("[02] Computing forensic profiles for Dorabella, Liszt (std/alt), and Marco (39/47)...")
    profiles["Dorabella_1897"] = doc_profile("Dorabella_1897", docs["Dorabella_1897"]["tokens"], rng)
    profiles["Liszt_1886_std"] = doc_profile("Liszt_1886_std", docs["Liszt_1886_std"]["tokens"], rng)
    profiles["Liszt_1886_alt"] = doc_profile("Liszt_1886_alt", docs["Liszt_1886_alt"]["tokens"], rng)
    profiles["Marco_1924_39"] = doc_profile("Marco_1924_39", docs["Marco_Notebook_1924"]["tokens_39"], rng)
    profiles["Marco_1924_47"] = doc_profile("Marco_1924_47", docs["Marco_Notebook_1924"]["tokens_47"], rng)

    d_vec = [profiles["Dorabella_1897"]["ori_counts"][c] for c in ORI_LIST]
    l_std_vec = [profiles["Liszt_1886_std"]["ori_counts"][c] for c in ORI_LIST]
    l_alt_vec = [profiles["Liszt_1886_alt"]["ori_counts"][c] for c in ORI_LIST]
    m39_vec = [profiles["Marco_1924_39"]["ori_counts"][c] for c in ORI_LIST]

    comparisons = {
        "Dorabella_vs_Liszt_std_ori": corr_and_perm(d_vec, l_std_vec, rng),
        "Dorabella_vs_Liszt_alt_ori": corr_and_perm(d_vec, l_alt_vec, rng),
        "Dorabella_vs_Marco39_ori": corr_and_perm(d_vec, m39_vec, rng),
        "Fisher_W_Dorabella_vs_Marco39": {
            "p": round(float(fisher_exact([[61, 26], [16, 23]])[1]), 6)
        },
        "Fisher_W_Liszt_std_vs_Marco39": {
            "p": round(float(fisher_exact([[16, 2], [16, 23]])[1]), 6)
        },
        "Chi2_arcs_Dorabella_vs_Liszt_std": {
            "p": round(float(chi2_contingency([profiles["Dorabella_1897"]["arc_counts"],
                                               profiles["Liszt_1886_std"]["arc_counts"]])[1]), 5)
        },
        "Chi2_arcs_Dorabella_vs_Marco39": {
            "p": round(float(chi2_contingency([profiles["Dorabella_1897"]["arc_counts"],
                                               profiles["Marco_1924_39"]["arc_counts"]])[1]), 5)
        }
    }

    print("[02] Running 500 Marco-key English controls (N=87 and N=18)...")
    l2t = docs["Marco_Notebook_1924"]["letter_to_token"]
    ctrl87 = marco_key_controls(l2t, 87, rng, n_ctrl=500)
    ctrl18 = marco_key_controls(l2t, 18, rng, n_ctrl=500)

    out = {
        "profiles": profiles,
        "pairwise_tests": comparisons,
        "marco_key_controls_N87": ctrl87,
        "marco_key_controls_N18": ctrl18
    }
    out_path = os.path.join(DATA_DIR, "results_02_corpus.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"  -> Dorabella vs Liszt_std: r={comparisons['Dorabella_vs_Liszt_std_ori']['pearson_r']} "
          f"(p={comparisons['Dorabella_vs_Liszt_std_ori']['p_two_tailed']}, cos={comparisons['Dorabella_vs_Liszt_std_ori']['cosine']})")
    print(f"  -> Dorabella vs Marco39  : r={comparisons['Dorabella_vs_Marco39_ori']['pearson_r']}, "
          f"Fisher W% p={comparisons['Fisher_W_Dorabella_vs_Marco39']['p']}, "
          f"Chi2 arcs p={comparisons['Chi2_arcs_Dorabella_vs_Marco39']['p']}")
    print(f"  -> Marco-key EN controls (N=87, n=500): W={ctrl87['pct_W_mean']}±{ctrl87['pct_W_sd']}%, "
          f"Z_WV={ctrl87['z_wv_mean']}±{ctrl87['z_wv_sd']} (0/500 <= -4.76), "
          f"180°={ctrl87['flips_180_mean']}±{ctrl87['flips_180_sd']} ({ctrl87['count_180_ge_27']}/500 >= 27)")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
