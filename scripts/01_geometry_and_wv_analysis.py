#!/usr/bin/env python3
"""
01_geometry_and_wv_analysis.py
==============================
Reproducible 2D geometric & dynamical analysis of the Dorabella Cipher (1897).

Computes:
  1. W={B,C,F,G} vs V={A,D,E,H} orientation-block segregation across all 35
     4+4 orientation partitions and calibrated against 1,000 English MASC controls.
  2. Two-layer decomposition of 180-degree opposite transitions (27/86 all-arc,
     13/27 same-arc vs 14/59 diff-arc, inside vs outside Massey's 3 non-repeating
     arc runs, and conditional intra-W/V permutations preserving both W/V blocks
     and arc positions).
  3. Exact exhaustive 105-pairing test of 180-degree arc-profile symmetry across
     the four opposite geometric axes (A<->E, B<->F, C<->G, D<->H).
  4. Autocorrelation lag=31 localization on the second half of Lines 1 & 2.

Outputs:
  ../data/results_01_geometry.json
"""

import json
import os
from itertools import combinations
from collections import Counter
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ORI_LIST = list("ABCDEFGH")
ORI_MAP = {c: i for i, c in enumerate(ORI_LIST)}
W_SET = {1, 2, 5, 6}  # B, C, F, G

# Authentic English literary sample (for 1,000 MASC length-87 controls)
ENGLISH_SAMPLE = (
    "ITISATRUTHUNIVERSALLYACKNOWLEDGEDTHATASINGLEMANINPOSSESSIONOFAGOODFORTUNEMUSTBEINWANTOFAWIFE"
    "HOWEVERLITTLEKNOWNTHEFEELINGSORVIEWSOFSUCHAMANMAYBEONHISFIRSTENTERINGANEIGHBOURHOODTHISTRUTH"
    "ISSOWELLFIXEDINTHEMINDSOFTHESURROUNDINGFAMILIESTHATHEISCONSIDEREDTHERIGHTFULPROPERTYOFSONE"
    "OROTHEROFTHEIRDAUGHTERSMYDEARMRBENNETSAIDHISLADYTOHIMONEDAYHAVEYOUHEARDTHATNETHERFIELDPARK"
    "ISLETATLASTMRBENNETREPLIEDTHATHEHADNOTBUTITISSHEANSWEREDFORMRSLONGHASJUSTBEENHEREANDSHETOLD"
    "MEALLABOUTITMRBENNETMADENOANSWERDONOTYOUWANTTOKNOWWHOHASTAKENITCRIEDHISWIFEIMPATIENTLYYOU"
    "WANTTOTELLMEANDIHAVENOOBJECTIONTOHEARINGITTHISWASINVITATIONENOUGHWHYMYDEARYOUMUSTKNOWMRSLONG"
    "SAYSTHATNETHERFIELDISAKENBYAYOUNGMANOFLARGEFORTUNEFROMTHENORTHOFENGLANDTHATHECAMEONMONDAY"
    "INACHAISEANDFOURTOSEETHEPLACEANDWASSOMUCHDELIGHTEDWITHITTHATHEAGREEDWITHMRMORRISIMMEDIATELY"
    "THATHEISTOTAKEPOSSESSIONBEFOREMICHAELMASANDSOMEOFHISSERVANTSARETOBEINTHEHOUSEBYTHEENDOFNEXT"
    "WEEKWHATISHISNAMEBINGLEYISHEMARRIEDORSINGLEOHSINGLEMYDEARTBESUREASINGLEMANOFLARGEFORTUNE"
    "FOURORFIVETHOUSANDAYEARWHATAFINETHINGFOROURGIRLSHOWSOCANITAFFECTTHEMMYDEARMRBENNETREPLIED"
    "HISWIFEHOWCANYOUBESOTIRESOMEYOUMUSTKNOWTHATIAMTHINKINGOFHISMARRYINGONEOFTHEMISSTHATHISDESIGN"
    "INSETTLINGHEREDESIGNNONSENSEHOWCANYOUTALKSOUTITISVERYLIKELYTHATHEMAYFALLINLOVEWITHONEOFTHEM"
    "ANDTHEREFOREYOUMUSTVISITHIMASSOONASHECOMESISEENOOCCASIONFORTHATYOUANDTHEGIRLSMAGYOROUMAY"
    "SENDTHEMBYTHEMSELVESWHICHPERHAPSWILLBESTILLBETTERFORASYOUAREASHANDSOMEASANYOFTHEMMRBINGLEY"
    "MIGHTLIKEYOUTHEBESTAOFTHEPARTYMYDEARYOUFLATTERMEICERTAINLYHAVEHADMYSHAREOFBEAUTYBUTIDONOT"
    "PRETENDTOBEANYTHINGEXTRAORDINARYNOWWHENAWOMANHASFIVEGROWNPDAUGHTERSHEOUGHTTOGIVEOVERTHINKING"
    "OFHEROWNBEAUTYINSUCHCASESAWOMANHASNOTOFTENMUCHBEAUTYTOTHINKOFBUTMYDEARYOUMUSTINGEDGOTOSEE"
    "MRBINGLEYWHENHECOMESINTOTHENEIGHBOURHOODITISMORETHANICANENGAGEFORIASSUREYOUBUTCONSIDER"
    "YOURDAUGHTERSONLYTHINKWHATANESTABLISHMENTITWOULDBEFORONEOFTHEMSIRWILLIAMANDLADYLUCASARE"
    "DETERMINEDTOGOMERELYONTHATACCOUNTFORINGENERALYOUKNOWTHEYVISITNONEWCOMERSINDEEDYOUMUSTGO"
    "FORITWILLBEIMPOSSIBLEFORUSTOVISITHIMIFYOUDONOTYOUAREOVERSCRUPULOUSSURELYIDARESAYMRBINGLEY"
    "WILLBEVERYGLADTOSEEYOUANDIWILLSENDAFEWLINESBYYOUTOASSUREHIMOFMYHEARTYCONSENTTOHISMARRYING"
    "WHICHEVERHECHUSESOFTHEGIRLSTHOUGHIMUSTTHROWINGAWORDGOODFORMYLITTLELIZZYIDESIREYOUWILLDO"
    "NOSUCHTHINGLIZZYISNOTABITBETTERTHANTHEOTHERSANDIAMSHURESHEISNOTHALFSOHANDSOMEASJANENOR"
    "HALFSOGOODHUMOUREDASLYDIABUTYOUAREALWAYSGIVINGHERTHEPREFERENCETHEYHAVENONEOFTHEMMUCHTO"
    "RECOMMENDTHEMREPLIEDHEHEYAREALLSILLYANDIGNORANTLIKEOTHERGIRLSBUTLIZZYHASSOMETHINGMOREOF"
    "QUICKNESSTHANHERSISTERSMRBENNETHOWCANYOUABUSEYOUROWNCHILDRENINSUCHAWAYYOUTAKEDELIGHTIN"
    "VEXINGMEYOUHAVENOCOMPASSIONFORNYPOORNERVESYOUIMISTAKEEMEDEARIAHVEAHIGHRESPECTFORYOURNERVES"
    "THEYAREMYOLDFRIENDSIHAVEHEARDYOUMENTIONTHEMWITHCONSIDERATIONTHESETWENTYYEARSATLEASTAHYOU"
    "DONOTKNOWWHATISUFFEBUTIHOPEDEARWEWILLGETOVERITANDLIVETOSEEMANYOUNGMANOFFOURTHOUSANDAYEAR"
    "COMEINTOTHENEIGHBOURHOODITWILLBENOUSETOUSIFTWENTYSUCHSHOULDCOMESINCEYOUWILLNOTVISITTHEM"
    "DEPENDUPONITMYDEARTHATWHENTHEREARETWENTYIWILLVISITTHEMALLMRBENNETWASSOODDAMIXXTUREOFQUICK"
    "PARTSARCASTICHUMOURRESERVEANDCAPRICETHATTHEEXPERIENCEOFTHREEANDTWENTYYEARSHADBEENINSUFFICIENT"
    "TOMAKEHISWIFEUNDERSTANDHISCHARACTERHERMINDWASLESSDIFFICULTTOBEVELOPSHEWASAWOMANOFMEAN"
    "UNDERSTANDINGLITTLEINFORMATIONANDUNCERTAINTEMPERWHENSHEDISCONTENTEDSHEFANCIEDHERSELFNERVOUS"
    "THEBUSINESSOFHERLIFEWASTOGETHERDAUGHTERSMARRIEDITSSOLACEWASVISITINGANDNEWSINANOTHERLETTER"
    "DEARELGARWROTEFROMFORLIMALVERNTODORAPENNYABOUTLOCOMOTIVELEARNINGANDWARBLINGWIGOROUSLYIN"
    "WORCESTERWUNCEAWEEKALLITERATIONARCHAICALLYENORSEANDSHOWINGHEROVERTHECATHEDRALORGANANDTHE"
    "DANESSKINANDTAKINGTODIESINKINGASARECREATIONSOHISMUSICMANUSCRIPTSMAYBEEASILYDISTINGUISHED"
)


def compute_ic(seq):
    n = len(seq)
    c = Counter(seq)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def exact_wv_zscore(oris, subset):
    """Exact analytical expectation and variance of crossings under random permutation."""
    n = len(oris)
    x = np.array([1 if o in subset else 0 for o in oris], dtype=int)
    n1 = int(x.sum())
    n0 = n - n1
    obs = int(np.sum(x[:-1] != x[1:]))
    if n1 == 0 or n0 == 0:
        return obs, 0.0, 0.0, 0.0
    # Under random permutation of n binary items (n1 ones, n0 zeros):
    # Number of transitions (crossings) = R - 1, where R is the Wald-Wolfowitz run count.
    # E[R - 1] = 2 * n1 * n0 / n
    # Var[R - 1] = 2*n1*n0*(2*n1*n0 - n) / (n^2 * (n - 1))
    exp = 2.0 * n1 * n0 / n
    var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n * n * (n - 1))
    sd = np.sqrt(max(var, 1e-12))
    z = (obs - exp) / sd
    return obs, float(exp), float(sd), float(z)


def all_pairings(items):
    """Generate all unique pairings of an even-sized list into unordered pairs (105 for n=8)."""
    if len(items) == 0:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        pair = (first, items[i])
        rest = items[1:i] + items[i + 1:]
        for sub in all_pairings(rest):
            yield [pair] + sub


def analyze_transcription(name, toks, rng, n_mc=50000):
    oris = np.array([ORI_MAP[t[0]] for t in toks])
    arcs = np.array([int(t[1]) for t in toks])
    is_W = np.array([o in W_SET for o in oris])
    lines = [toks[:29], toks[29:60], toks[60:]]
    offs = [0, 29, 60]

    # Basic stats
    ic = compute_ic(toks)
    dbl = int(sum(toks[i] == toks[i + 1] for i in range(len(toks) - 1)))
    bigrams = Counter(zip(toks[:-1], toks[1:]))
    rev_d = int(sum(1 for (a, b) in bigrams if a < b and (b, a) in bigrams))

    # Orientation x Arc chi-squared
    mat = np.zeros((8, 3), dtype=int)
    for o, a in zip(oris, arcs):
        mat[o, a - 1] += 1
    chi2_oa, p_oa, _, _ = chi2_contingency(mat)

    # W/V crossings
    cross_obs, cross_exp, cross_sd, z_wv = exact_wv_zscore(oris, W_SET)

    # All 35 unique 4+4 partitions (fix 0 in first half to avoid complements)
    part_results = []
    for rest in combinations(range(1, 8), 3):
        s = {0, *rest}
        c_obs, c_exp, c_sd, c_z = exact_wv_zscore(oris, s)
        lbl = "".join(ORI_LIST[i] for i in sorted(s))
        comp = "".join(ORI_LIST[i] for i in range(8) if i not in s)
        part_results.append({
            "group_a": lbl, "group_b": comp,
            "crossings": c_obs, "expected": round(c_exp, 2), "z": round(c_z, 3)
        })
    part_results.sort(key=lambda d: d["z"])

    # F-to-V contact count
    f_contacts = []
    for i, o in enumerate(oris):
        if o == 5:  # F
            if i > 0:
                f_contacts.append(oris[i - 1])
            if i < len(oris) - 1:
                f_contacts.append(oris[i + 1])
    f_to_v = int(sum(1 for c in f_contacts if c not in W_SET))
    mc_fv = np.zeros(n_mc)
    for k in range(n_mc):
        po = rng.permutation(oris)
        cnt = 0
        for i, o in enumerate(po):
            if o == 5:
                if i > 0 and po[i - 1] not in W_SET:
                    cnt += 1
                if i < 86 and po[i + 1] not in W_SET:
                    cnt += 1
        mc_fv[k] = cnt

    # Massey longest non-repeating arc runs per line
    m_trans = set()
    run_info = []
    for li, (lt, off) in enumerate(zip(lines, offs), 1):
        bs, bl, cs = 0, 1, 0
        for j in range(1, len(lt)):
            if lt[j][1] != lt[j - 1][1]:
                if j - cs + 1 > bl:
                    bl, bs = j - cs + 1, cs
            else:
                cs = j
        run_info.append({
            "line": li, "start_1based": bs + 1, "end_1based": bs + bl,
            "length_glyphs": bl, "length_transitions": bl - 1
        })
        m_trans.update(range(off + bs, off + bs + bl - 1))

    # 180-degree transitions breakdown (86 global, 84 intra-line)
    intra = np.ones(86, dtype=bool)
    intra[28] = False
    intra[59] = False
    t180 = ((oris[:-1] - oris[1:]) % 8 == 4)
    tsam = (arcs[:-1] == arcs[1:])
    tdiff = ~tsam
    tinm = np.array([i in m_trans for i in range(86)])

    c_all = int(t180.sum())
    c_all_intra = int((t180 & intra).sum())
    c_same = int((t180 & tsam).sum())
    n_same = int(tsam.sum())
    c_diff = int((t180 & tdiff).sum())
    n_diff = int(tdiff.sum())
    c_inm = int((t180 & tinm).sum())
    n_inm = int(tinm.sum())
    c_outm = c_all - c_inm
    n_outm = 86 - n_inm

    or_sd, p_fish_sd = fisher_exact([[c_same, n_same - c_same], [c_diff, n_diff - c_diff]])
    or_io, p_fish_io = fisher_exact([[c_inm, n_inm - c_inm], [c_outm, n_outm - c_outm]])

    # Permutations (free token, ori-only, intra-W/V preserving arcs)
    w_idx, v_idx = np.where(is_W)[0], np.where(~is_W)[0]
    w_o, v_o = oris[w_idx], oris[v_idx]

    mc_all = np.zeros(n_mc)
    mc_same = np.zeros(n_mc)
    mc_diff = np.zeros(n_mc)
    mc_ori_diff = np.zeros(n_mc)
    mc_ori_inm = np.zeros(n_mc)
    mc_wv_all = np.zeros(n_mc)
    mc_wv_same = np.zeros(n_mc)
    mc_wv_diff = np.zeros(n_mc)
    mc_wv_inm = np.zeros(n_mc)

    for k in range(n_mc):
        p = rng.permutation(87)
        po_f, pa_f = oris[p], arcs[p]
        tt = ((po_f[:-1] - po_f[1:]) % 8 == 4)
        ts = (pa_f[:-1] == pa_f[1:])
        mc_all[k] = tt.sum()
        mc_same[k] = (tt & ts).sum()
        mc_diff[k] = (tt & ~ts).sum()

        po = rng.permutation(oris)
        tt2 = ((po[:-1] - po[1:]) % 8 == 4)
        mc_ori_diff[k] = (tt2 & tdiff).sum()
        mc_ori_inm[k] = (tt2 & tinm).sum()

        po3 = np.empty(87, dtype=int)
        po3[w_idx] = rng.permutation(w_o)
        po3[v_idx] = rng.permutation(v_o)
        tt3 = ((po3[:-1] - po3[1:]) % 8 == 4)
        mc_wv_all[k] = tt3.sum()
        mc_wv_same[k] = (tt3 & tsam).sum()
        mc_wv_diff[k] = (tt3 & tdiff).sum()
        mc_wv_inm[k] = (tt3 & tinm).sum()

    def stat_block(obs, arr):
        return {
            "obs": int(obs),
            "exp_mean": round(float(arr.mean()), 3),
            "exp_sd": round(float(arr.std()), 3),
            "z": round(float((obs - arr.mean()) / (arr.std() + 1e-12)), 3),
            "p_ge": round(float(np.mean(arr >= obs)), 6)
        }

    # Exact 105-pairing 180-degree arc-profile symmetry test
    row_sums = mat.sum(axis=1, keepdims=True)
    P = mat / np.maximum(row_sums, 1)
    mean_arc = (mat @ np.array([1.0, 2.0, 3.0])) / np.maximum(mat.sum(axis=1), 1)
    pairings = list(all_pairings(list(range(8))))
    assert len(pairings) == 105

    def pairing_scores(pairs):
        cos_s, diff_s = 0.0, 0.0
        for i, j in pairs:
            a, b = P[i], P[j]
            cos_s += float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            diff_s += abs(float(mean_arc[i] - mean_arc[j]))
        return cos_s / 4.0, diff_s / 4.0

    opp_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    obs_cos, obs_adiff = pairing_scores(opp_pairs)
    all_cos = [pairing_scores(pr)[0] for pr in pairings]
    all_adiff = [pairing_scores(pr)[1] for pr in pairings]
    rank_cos = int(sum(1 for v in all_cos if v >= obs_cos - 1e-12))
    rank_adiff = int(sum(1 for v in all_adiff if v <= obs_adiff + 1e-12))

    # Lag=31 localization
    lag31_matches = [i for i in range(87 - 31) if toks[i] == toks[i + 31]]
    l1_l2_early = [i for i in range(0, 17) if toks[i] == toks[i + 31]]
    l1_l2_late = [i for i in range(17, 29) if toks[i] == toks[i + 31]]
    l2_l3_matches = [i for i in range(29, 56) if toks[i] == toks[i + 31]]

    return {
        "name": name,
        "basic": {
            "n_tokens": len(toks),
            "unique_symbols": len(set(toks)),
            "ic": round(ic, 5),
            "doubles": dbl,
            "reversed_digrams": rev_d,
            "chi2_ori_arc": round(float(chi2_oa), 3),
            "p_ori_arc": round(float(p_oa), 6),
            "ori_arc_matrix": {ORI_LIST[i]: mat[i].tolist() for i in range(8)},
            "mean_arc_by_ori": {ORI_LIST[i]: round(float(mean_arc[i]), 3) for i in range(8)}
        },
        "wv_segregation": {
            "n_W": int(is_W.sum()),
            "pct_W": round(100.0 * float(is_W.mean()), 2),
            "n_V": int((~is_W).sum()),
            "crossings": cross_obs,
            "expected": round(cross_exp, 3),
            "sd": round(cross_sd, 3),
            "z_wv": round(z_wv, 3),
            "top5_partitions": part_results[:5],
            "all_35_z_values": [d["z"] for d in part_results],
            "f_to_v_contacts": {
                "obs": f_to_v,
                "total_f_contacts": len(f_contacts),
                "exp_mean": round(float(mc_fv.mean()), 2),
                "p_le": round(float(np.mean(mc_fv <= f_to_v)), 6)
            }
        },
        "massey_runs": run_info,
        "flips_180": {
            "all_86": c_all,
            "intra_84": c_all_intra,
            "same_arc": {"count": c_same, "total": n_same, "pct": round(100.0 * c_same / n_same, 2)},
            "diff_arc": {"count": c_diff, "total": n_diff, "pct": round(100.0 * c_diff / n_diff, 2)},
            "in_massey_runs": {"count": c_inm, "total": n_inm, "pct": round(100.0 * c_inm / n_inm, 2)},
            "out_massey_runs": {"count": c_outm, "total": n_outm, "pct": round(100.0 * c_outm / n_outm, 2)},
            "fisher_same_vs_diff": {"odds_ratio": round(float(or_sd), 3), "p": round(float(p_fish_sd), 5)},
            "fisher_in_vs_out_massey": {"odds_ratio": round(float(or_io), 3), "p": round(float(p_fish_io), 5)},
            "perm_free_token": {
                "all_180": stat_block(c_all, mc_all),
                "same_arc_180": stat_block(c_same, mc_same),
                "diff_arc_180": stat_block(c_diff, mc_diff)
            },
            "perm_ori_fixed_arcs": {
                "diff_arc_180": stat_block(c_diff, mc_ori_diff),
                "in_massey_180": stat_block(c_inm, mc_ori_inm)
            },
            "perm_intra_wv_fixed_arcs": {
                "all_180": stat_block(c_all, mc_wv_all),
                "same_arc_180": stat_block(c_same, mc_wv_same),
                "diff_arc_180": stat_block(c_diff, mc_wv_diff),
                "in_massey_180": stat_block(c_inm, mc_wv_inm)
            }
        },
        "arc_symmetry_180_exact105": {
            "obs_mean_cosine": round(obs_cos, 4),
            "null_mean_cosine": round(float(np.mean(all_cos)), 4),
            "null_sd_cosine": round(float(np.std(all_cos)), 4),
            "exact_rank_cosine": rank_cos,
            "exact_p_cosine": round(rank_cos / 105.0, 5),
            "obs_mean_abs_diff": round(obs_adiff, 4),
            "null_mean_abs_diff": round(float(np.mean(all_adiff)), 4),
            "exact_rank_abs_diff": rank_adiff,
            "exact_p_abs_diff": round(rank_adiff / 105.0, 5)
        },
        "lag31_localization": {
            "total_matches_of_56": len(lag31_matches),
            "l1_l2_early_17": len(l1_l2_early),
            "l1_l2_late_12": len(l1_l2_late),
            "l2_l3_27": len(l2_l3_matches),
            "l1_late_slice": toks[17:29],
            "l2_late_slice": toks[48:60],
            "l1_l2_late_matched_tokens": [
                {"l1_pos": i + 1, "l2_pos": (i + 31 - 29) + 1, "token": toks[i]}
                for i in l1_l2_late
            ]
        }
    }


def calibrate_min_z_english_masc(rng, n_ctrl=1000):
    """Calibrate minimum Z_WV across all 35 4+4 partitions on 1,000 random EN MASC controls."""
    parts = [{0, *rest} for rest in combinations(range(1, 8), 3)]
    # Map 26 English letters randomly onto 24 (ori, arc) symbols (merging J=I, V=U)
    alpha24 = list("ABCDEFGHIKLMNOPQRSTUWXYZ")
    n_text = len(ENGLISH_SAMPLE)
    min_zs = np.zeros(n_ctrl)
    for k in range(n_ctrl):
        st = int(rng.integers(0, n_text - 87))
        sub = ENGLISH_SAMPLE[st:st + 87].replace("J", "I").replace("V", "U")
        perm_syms = rng.permutation(24)
        l2o = {alpha24[i]: int(perm_syms[i] // 3) for i in range(24)}
        oris = np.array([l2o[c] for c in sub])
        min_zs[k] = min(exact_wv_zscore(oris, p)[3] for p in parts)
    return {
        "n_controls": n_ctrl,
        "mean_min_z": round(float(min_zs.mean()), 3),
        "sd_min_z": round(float(min_zs.std()), 3),
        "worst_min_z": round(float(min_zs.min()), 3),
        "count_le_dorabella": int(np.sum(min_zs <= -4.756)),
        "p_corrected_upper": round(float((np.sum(min_zs <= -4.756) + 1) / (n_ctrl + 1)), 5)
    }


def main():
    rng = np.random.default_rng(20260714)
    with open(os.path.join(DATA_DIR, "transcriptions.json")) as f:
        tdata = json.load(f)

    results = {"transcriptions": {}}
    for name in ["Hauer_2025", "Hartmeier_2017", "Pelling_2012"]:
        toks = tdata["transcriptions"][name]["tokens"]
        print(f"[01] Analyzing {name} (50,000 permutations + exact 105 pairings)...")
        results["transcriptions"][name] = analyze_transcription(name, toks, rng, n_mc=50000)

    print("[01] Calibrating look-elsewhere min(Z_WV) on 1,000 English MASC controls...")
    results["en_masc_min_z_calibration"] = calibrate_min_z_english_masc(rng, n_ctrl=1000)

    out_path = os.path.join(DATA_DIR, "results_01_geometry.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    h = results["transcriptions"]["Hauer_2025"]
    print(f"  -> Hauer_2025: Z_WV={h['wv_segregation']['z_wv']} ({h['wv_segregation']['crossings']}/86), "
          f"180°={h['flips_180']['all_86']}/86 (same={h['flips_180']['same_arc']['count']}/27, "
          f"diff={h['flips_180']['diff_arc']['count']}/59, inMassey={h['flips_180']['in_massey_runs']['count']}/29), "
          f"ArcSym180 cos={h['arc_symmetry_180_exact105']['obs_mean_cosine']} "
          f"(rank {h['arc_symmetry_180_exact105']['exact_rank_cosine']}/105, p={h['arc_symmetry_180_exact105']['exact_p_cosine']})")
    print(f"  -> EN MASC 1000-ctrl min(Z_WV): mean={results['en_masc_min_z_calibration']['mean_min_z']}, "
          f"worst={results['en_masc_min_z_calibration']['worst_min_z']}, "
          f"count<=-4.76={results['en_masc_min_z_calibration']['count_le_dorabella']}/1000")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
