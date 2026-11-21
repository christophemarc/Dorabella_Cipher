#!/usr/bin/env python3
"""
04_generate_figures.py
======================
Generates 4 publication-grade 300 DPI figures in ../figures/ directly from
the reproducible JSON outputs in ../data/.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

NAVY = "#1E3A5F"
TEAL = "#0F766E"
TEAL_LIGHT = "#CCFBF1"
AMBER = "#C2410C"
AMBER_LIGHT = "#FFEDD5"
CRIMSON = "#BE123C"
PURPLE = "#6D28D9"
SLATE = "#475569"
GOLD = "#D97706"

ORI_LIST = list("ABCDEFGH")
ORI_MAP = {c: i for i, c in enumerate(ORI_LIST)}
W_SET = {1, 2, 5, 6}


def make_fig1(tdata, r1):
    """Figure 1: Complete 2D structural map of the 3 lines of Dorabella."""
    toks = tdata["transcriptions"]["Hauer_2025"]["tokens"]
    lines = [toks[:29], toks[29:60], toks[60:]]
    offs = [0, 29, 60]
    runs = r1["transcriptions"]["Hauer_2025"]["massey_runs"]
    l1_late_matches = {d["l1_pos"] - 1 for d in r1["transcriptions"]["Hauer_2025"]["lag31_localization"]["l1_l2_late_matched_tokens"]}
    l2_late_matches = {d["l2_pos"] - 1 for d in r1["transcriptions"]["Hauer_2025"]["lag31_localization"]["l1_l2_late_matched_tokens"]}

    fig, ax = plt.subplots(figsize=(15.8, 6.2), dpi=300)
    ax.set_xlim(-2.9, 31.8)
    ax.set_ylim(-0.7, 6.9)
    ax.axis("off")

    ax.text(14.5, 6.55,
            "Cartographie 2D du chiffre de Dorabella (Hauer et al., 2025 ; N = 87)",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color=NAVY)
    ax.text(14.5, 6.12,
            "Blocs de posture W={B,C,F,G} vs V={A,D,E,H} (Z = -4,76), séries sans répétition d'arcs (Massey), "
            "sauts à 180° (13 même-arc + 14 arcs différents) et parallélisme lag=31 en fin de L1/L2",
            ha="center", va="center", fontsize=9.2, color=SLATE)

    y_rows = [4.65, 2.75, 0.85]
    tile_w, tile_h = 0.84, 0.64

    for li, (line, off, y0) in enumerate(zip(lines, offs, y_rows)):
        # Line label
        nw = sum(1 for t in line if ORI_MAP[t[0]] in W_SET)
        ax.text(-0.68, y0 + 0.10, f"Ligne {li+1}", ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=NAVY)
        ax.text(-0.68, y0 - 0.22, f"({len(line)} sym, W={100*nw/len(line):.0f}%)",
                ha="right", va="center", fontsize=8.0, color=SLATE)

        # Draw lag=31 late box for L1 (#18..29 -> idx 17..28) and L2 (#20..31 -> idx 19..30)
        if li == 0:
            rect = patches.FancyBboxPatch((16.50, y0 - 0.44), 12.02, 0.90,
                                          boxstyle="round,pad=0.04,rounding_size=0.12",
                                          facecolor="none", edgecolor=GOLD, linewidth=1.6, linestyle="--", zorder=1)
            ax.add_patch(rect)
            ax.text(22.5, y0 - 0.60, "Fin L1 alignée à lag=31 avec fin L2 (6/12 exacts : F2 C2 ... B1 H1 ... B3 F3)",
                    ha="center", va="top", fontsize=7.6, color="#B45309", fontweight="bold")
        elif li == 1:
            rect = patches.FancyBboxPatch((18.50, y0 - 0.44), 12.02, 0.90,
                                          boxstyle="round,pad=0.04,rounding_size=0.12",
                                          facecolor="none", edgecolor=GOLD, linewidth=1.6, linestyle="--", zorder=1)
            ax.add_patch(rect)

        # Draw Massey non-repeating arc bracket below tiles
        rm = runs[li]
        xs = rm["start_1based"] - 1 - 0.42
        xe = rm["end_1based"] - 1 + 0.42
        yb = y0 - 0.48
        ax.plot([xs, xs, xe, xe], [yb + 0.09, yb, yb, yb + 0.09], color=NAVY, lw=1.5, zorder=2)
        ax.text((xs + xe) / 2, yb - 0.14,
                f"Série Massey sans rép. d'arcs ({rm['length_glyphs']} glyphes)",
                ha="center", va="top", fontsize=7.5, color=NAVY, style="italic")

        # Draw tiles
        for j, tok in enumerate(line):
            is_w = ORI_MAP[tok[0]] in W_SET
            fc = TEAL_LIGHT if is_w else AMBER_LIGHT
            ec = TEAL if is_w else AMBER
            is_lag_hit = (li == 0 and j in l1_late_matches) or (li == 1 and j in l2_late_matches)
            lw = 2.0 if is_lag_hit else 1.0
            box = patches.FancyBboxPatch((j - tile_w / 2, y0 - tile_h / 2), tile_w, tile_h,
                                         boxstyle="round,pad=0.02,rounding_size=0.10",
                                         facecolor=fc, edgecolor=GOLD if is_lag_hit else ec,
                                         linewidth=lw, zorder=3)
            ax.add_patch(box)
            ax.text(j, y0 + 0.03, tok, ha="center", va="center",
                    fontsize=8.3, fontweight="bold", color="#0F172A", zorder=4)
            # Dot after L3#5
            if li == 2 and j == 4:
                ax.scatter([j + 0.49], [y0 - 0.22], s=26, color="#0F172A", zorder=6)

        # Draw 180-degree transition arcs above adjacent pairs
        for j in range(len(line) - 1):
            o1, a1 = ORI_MAP[line[j][0]], int(line[j][1])
            o2, a2 = ORI_MAP[line[j + 1][0]], int(line[j + 1][1])
            if (o1 - o2) % 8 == 4:
                same = (a1 == a2)
                color = CRIMSON if same else PURPLE
                ls = "-" if same else "--"
                arc = patches.Arc((j + 0.5, y0 + 0.34), width=0.95, height=0.58,
                                  angle=0, theta1=0, theta2=180, color=color,
                                  linewidth=1.8 if same else 1.5, linestyle=ls, zorder=5)
                ax.add_patch(arc)
                ax.text(j + 0.5, y0 + 0.72, "S" if same else "D",
                        ha="center", va="center", fontsize=6.8, fontweight="bold", color=color)

    # Legend strip at bottom
    leg_y = -0.35
    legend_items = [
        patches.Patch(facecolor=TEAL_LIGHT, edgecolor=TEAL, label="Posture W = {B, C, F, G} (ondes, 70,1%)"),
        patches.Patch(facecolor=AMBER_LIGHT, edgecolor=AMBER, label="Posture V = {A, D, E, H} (vert./diag., 29,9%)"),
        plt.Line2D([0], [0], color=CRIMSON, lw=2.0, ls="-", label="Saut 180° même arc (S : 13/27 = 48,1%)"),
        plt.Line2D([0], [0], color=PURPLE, lw=1.7, ls="--", label="Saut 180° arcs différents (D : 13 intra + 1 L2→L3 = 14/59)"),
        patches.Patch(facecolor="none", edgecolor=GOLD, lw=1.6, ls="--", label="Squelette lag=31 fin L1/L2 (6/12 identiques)"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_dorabella_2d_map.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved {out}")


def make_fig2(r1):
    """Figure 2: Three core 2D geometric signatures of Dorabella."""
    h = r1["transcriptions"]["Hauer_2025"]
    cal = r1["en_masc_min_z_calibration"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300)

    # --- Panel A: All 35 4+4 partitions Z_WV ---
    ax = axes[0]
    zs = sorted(h["wv_segregation"]["all_35_z_values"])
    colors = [CRIMSON if i == 0 else "#94A3B8" for i in range(len(zs))]
    ax.bar(range(1, 36), zs, color=colors, edgecolor="none", width=0.8)
    ax.axhline(0, color="#334155", lw=0.8)
    ax.axhline(cal["worst_min_z"], color=NAVY, lw=1.3, ls="--",
               label=f"Pire de 1 000 MASC EN (Z={cal['worst_min_z']:.2f})")
    ax.annotate("W={B,C,F,G} vs V={A,D,E,H}\nZ = -4,76 (18/86 crois.)\n0 / 1 000 MASC EN",
                xy=(1, zs[0]), xytext=(6.5, -4.35),
                fontsize=8.2, fontweight="bold", color=CRIMSON,
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF1F2", ec=CRIMSON, lw=0.9))
    ax.set_title("A. Ségrégation W/V parmi les 35 partitions 4+4")
    ax.set_xlabel("Rang parmi les 35 partitions 4+4 d'orientations")
    ax.set_ylabel("Score Z des transitions inter-groupes")
    ax.set_ylim(-5.3, 3.1)
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white")

    # --- Panel B: 180-degree ergonomic arc-profile symmetry ---
    ax = axes[1]
    mean_arc = h["basic"]["mean_arc_by_ori"]
    axis_labels = ["Vertical droit\nA (0°) ↔ E (180°)",
                   "Onde inclinée\nB (45°) ↔ F (225°)",
                   "Onde horiz.\nC (90°) ↔ G (270°)",
                   "Diagonale raide\nD (135°) ↔ H (315°)"]
    ori_a = [mean_arc[c] for c in ["A", "B", "C", "D"]]
    ori_b = [mean_arc[c] for c in ["E", "F", "G", "H"]]
    x = np.arange(4)
    w = 0.34
    b1 = ax.bar(x - w / 2, ori_a, width=w, color=NAVY, label="Orientation o (A, B, C, D)")
    b2 = ax.bar(x + w / 2, ori_b, width=w, color=TEAL, label="Opposée o+180° (E, F, G, H)")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1E293B")
    sym = h["arc_symmetry_180_exact105"]
    ax.text(0.5, 0.93,
            f"Cosinus moyen des 4 axes = {sym['obs_mean_cosine']:.4f}\n"
            f"Rang exact : #{sym['exact_rank_cosine']} / 105 appariements (p = {sym['exact_p_cosine']:.4f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.0, fontweight="bold",
            color=NAVY, bbox=dict(boxstyle="round,pad=0.32", fc="#EFF6FF", ec=NAVY, lw=0.8))
    ax.set_title("B. Symétrie ergonomique d'arcs à 180°")
    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, fontsize=8.0)
    ax.set_ylabel("Nombre moyen d'arcs (1 à 3)")
    ax.set_ylim(0, 3.85)
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=7.6, frameon=True, facecolor="white")

    # --- Panel C: Two-layer 180° decomposition ---
    ax = axes[2]
    f180 = h["flips_180"]
    cats = ["Dans séries\nMassey (29)", "Diff-arc\nglobal (59)", "Même-arc\n(Massey, 27)"]
    obs_pct = [f180["in_massey_runs"]["pct"], f180["diff_arc"]["pct"], f180["same_arc"]["pct"]]
    null_ori_pct = [
        100.0 * f180["perm_ori_fixed_arcs"]["in_massey_180"]["exp_mean"] / 29.0,
        100.0 * f180["perm_ori_fixed_arcs"]["diff_arc_180"]["exp_mean"] / 59.0,
        100.0 * f180["perm_free_token"]["same_arc_180"]["exp_mean"] / 27.0,
    ]
    null_wv_pct = [
        100.0 * f180["perm_intra_wv_fixed_arcs"]["in_massey_180"]["exp_mean"] / 29.0,
        100.0 * f180["perm_intra_wv_fixed_arcs"]["diff_arc_180"]["exp_mean"] / 59.0,
        100.0 * f180["perm_intra_wv_fixed_arcs"]["same_arc_180"]["exp_mean"] / 27.0,
    ]
    x = np.arange(3)
    w = 0.26
    r_a = ax.bar(x - w, null_ori_pct, width=w, color="#CBD5E1", edgecolor="#64748B", lw=0.7, label="Nul : perm. libre")
    r_b = ax.bar(x, null_wv_pct, width=w, color="#93C5FD", edgecolor=NAVY, lw=0.7, label="Nul : intra-blocs W/V")
    r_c = ax.bar(x + w, obs_pct, width=w, color=[PURPLE, PURPLE, CRIMSON], label="Observé (diff-arc / même-arc)")
    p_labels = [
        f"p={f180['perm_ori_fixed_arcs']['in_massey_180']['p_ge']:.3f}\n(W/V: {f180['perm_intra_wv_fixed_arcs']['in_massey_180']['p_ge']:.2f})",
        f"p={f180['perm_ori_fixed_arcs']['diff_arc_180']['p_ge']:.3f}\n(W/V: {f180['perm_intra_wv_fixed_arcs']['diff_arc_180']['p_ge']:.2f})",
        f"p={f180['perm_intra_wv_fixed_arcs']['same_arc_180']['p_ge']:.4f}\n(même intra-W/V!)"
    ]
    for xi, (val, pl) in enumerate(zip(obs_pct, p_labels)):
        ax.text(xi + w, val + 1.4, f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color=CRIMSON if xi == 2 else PURPLE)
        ax.text(xi, max(val, null_wv_pct[xi]) + 6.5, pl, ha="center", va="bottom",
                fontsize=7.1, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", lw=0.6))
    ax.set_title("C. Décomposition 2 couches des 27/86 sauts 180°")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.3)
    ax.set_ylabel("Taux de transitions à 180° (%)")
    ax.set_ylim(0, 66)
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=7.4, frameon=True, facecolor="white")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_geometry_signatures.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved {out}")


def make_fig3(r2):
    """Figure 3: Cross-document comparison across Elgar's surviving semicircle corpus."""
    profs = r2["profiles"]
    pw = r2["pairwise_tests"]
    ctrl = r2["marco_key_controls_N87"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), dpi=300)

    # --- Panel A: 8-orientation percentage profiles ---
    ax = axes[0]
    x = np.arange(8)
    # Shade W columns (B=1, C=2, F=5, G=6)
    for wi in [1, 2, 5, 6]:
        ax.axvspan(wi - 0.45, wi + 0.45, color=TEAL_LIGHT, alpha=0.45, zorder=0)
    w = 0.26
    v_d = [profs["Dorabella_1897"]["ori_pct"][c] for c in ORI_LIST]
    v_l = [profs["Liszt_1886_std"]["ori_pct"][c] for c in ORI_LIST]
    v_m = [profs["Marco_1924_39"]["ori_pct"][c] for c in ORI_LIST]

    ax.bar(x - w, v_l, width=w, color=TEAL, label=f"Liszt Fragment 1886 (N=18, W=88,9%)")
    ax.bar(x, v_d, width=w, color=NAVY, label=f"Dorabella 1897 (N=87, W=70,1%)")
    ax.bar(x + w, v_m, width=w, color=AMBER, label=f"Carnet Marco 1924+ (N=39, W=41,0%)")

    ax.text(0.03, 0.95,
            f"Dorabella ↔ Liszt : r = +{pw['Dorabella_vs_Liszt_std_ori']['pearson_r']:.3f} "
            f"(p = {pw['Dorabella_vs_Liszt_std_ori']['p_two_tailed']:.3f}, cos = {pw['Dorabella_vs_Liszt_std_ori']['cosine']:.3f})\n"
            f"Dorabella ↔ Marco : r = {pw['Dorabella_vs_Marco39_ori']['pearson_r']:.3f} "
            f"(Fisher W% : p = {pw['Fisher_W_Dorabella_vs_Marco39']['p']:.4f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#94A3B8", lw=0.8))

    ax.set_title("A. Profils marginaux des 8 orientations (zones ombrées = classe W)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({i*45}°)" for i, c in enumerate(ORI_LIST)], fontsize=8.5)
    ax.set_ylabel("Fréquence (%)")
    ax.set_ylim(0, 36)
    ax.grid(axis="y")
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white")

    # --- Panel B: Dynamic phase space (Z_WV vs % 180° transitions) ---
    ax = axes[1]
    sc = ctrl["samples_for_plot"]
    ax.scatter(sc["z_wv"], sc["flips_180_pct"], s=20, alpha=0.42, color="#94A3B8",
               edgecolors="none", label="500 témoins EN chiffrés par clé Marco (N=87)")

    # Plot actual documents
    pts = [
        ("Carnet Marco 1924+\n(N=39, vrai texte EN)", profs["Marco_1924_39"]["z_wv"],
         100.0 * profs["Marco_1924_39"]["flips_180_all"] / 38.0, AMBER, "s", (12, -26)),
        ("Liszt Fragment 1886\n(N=18 : 0/17 sauts 180°)", profs["Liszt_1886_std"]["z_wv"],
         0.0, TEAL, "^", (-115, 16)),
        ("DORABELLA 1897\n(Z_WV = -4,76 ; 27/86 = 31,4%)", profs["Dorabella_1897"]["z_wv"],
         100.0 * profs["Dorabella_1897"]["flips_180_all"] / 86.0, CRIMSON, "*", (14, -18)),
    ]
    for lbl, px, py, col, mk, offset in pts:
        ms = 230 if mk == "*" else 110
        ax.scatter([px], [py], s=ms, color=col, marker=mk, edgecolors="#0F172A", linewidth=1.0, zorder=6)
        ax.annotate(lbl, xy=(px, py), xytext=offset, textcoords="offset points",
                    fontsize=8.2, fontweight="bold", color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, lw=0.9))

    ax.axvline(0, color="#64748B", lw=0.8, ls=":")
    ax.set_title("B. Espace dynamique (ségrégation W/V vs taux de sauts 180°)")
    ax.set_xlabel("Ségrégation en blocs W/V ($Z_{WV}$, plus négatif = plus ségrégué)")
    ax.set_ylabel("Transitions opposées à 180° (% de toutes les transitions)")
    ax.set_xlim(-5.5, 3.2)
    ax.set_ylim(-3.5, 38.0)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.9, frameon=True, facecolor="white")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_elgar_corpus_compare.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved {out}")


def make_fig4(r3):
    """Figure 4: LOO-CV model selection & Generative model competition heatmap."""
    loocv = r3["loocv_model_selection"]
    gen = r3["generative_competition"]

    fig, axes = plt.subplots(1, 2, figsize=(16.6, 5.1), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.62})

    # --- Panel A: LOO-CV Delta vs IID ---
    ax = axes[0]
    models_sorted = sorted(loocv, key=lambda d: d["delta_vs_iid"])
    labels = [f"{m['label']}\n(+{m['extra_params']} p.)" for m in models_sorted]
    deltas = [m["delta_vs_iid"] for m in models_sorted]
    cols = [CRIMSON if "Posture W/V + Réflexe" in m["label"] else (NAVY if m["delta_vs_iid"] > 0 else "#94A3B8")
            for m in models_sorted]
    y = np.arange(len(models_sorted))
    bars = ax.barh(y, deltas, color=cols, height=0.62, edgecolor="none")
    for bar, m in zip(bars, models_sorted):
        v = m["delta_vs_iid"]
        ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f} nats  (ppl={m['perplexity']:.2f})",
                va="center", ha="left", fontsize=8.0,
                fontweight="bold" if "Posture W/V + Réflexe" in m["label"] else "normal",
                color=CRIMSON if "Posture W/V + Réflexe" in m["label"] else "#1E293B")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.1)
    ax.set_xlabel("Gain de log-vraisemblance hors-échantillon LOO-CV vs IID (nats)")
    ax.set_title("A. Sélection de modèle (86 transitions d'orientation)")
    ax.set_xlim(0, 18.8)
    ax.grid(axis="x")

    # --- Panel B: Generative competition |Z| heatmap ---
    ax = axes[1]
    metric_keys = ["IC", "W_pct", "Z_WV", "flips_180_all", "flips_180_same", "chi2_OA", "max_no_rep_arc", "drift_W_pct"]
    metric_lbls = ["IC", "W%", "Z_WV", "180°\ntous", "180°\nmême", "χ²\nori×arc", "MaxNoRep\narcs", "Dérive\nW%"]
    gen_fr = {
        "G2_Calligraphic_calibrated": "G2 Calligraphique\n(W/V + miroir 2 couches)",
        "G1_Text_EN_MASC": "G1 Texte EN MASC\n(clé aléatoire)",
        "G3b_Elgar_Music_tetrachord_W": "G3b Musique Elgar\n(tétracorde W ad hoc)",
        "G3c_Elgar_Music_random_key": "G3c Musique Elgar\n(clé 8-ori aléatoire)",
        "G3a_Elgar_Music_clockwise": "G3a Musique Elgar\n(gamme horaire Atance)",
    }
    mods = sorted(gen["models"], key=lambda m: m["rms_z_capped20"])
    row_lbls = [f"{gen_fr[m['id']]}\n(RMS = {m['rms_z_capped20']:.2f}σ)" for m in mods]

    Z_mat = np.zeros((len(mods), len(metric_keys)))
    Z_disp = np.zeros((len(mods), len(metric_keys)))
    for i, m in enumerate(mods):
        for j, k in enumerate(metric_keys):
            zv = m["z_devs"][k]
            Z_disp[i, j] = zv
            Z_mat[i, j] = min(abs(zv), 6.0)

    im = ax.imshow(Z_mat, cmap="RdYlGn_r", vmin=0.0, vmax=6.0, aspect="auto")
    for i in range(len(mods)):
        for j in range(len(metric_keys)):
            zv = Z_disp[i, j]
            txt = "∞" if abs(zv) >= 50 else f"{zv:+.1f}σ"
            tcol = "white" if Z_mat[i, j] > 4.2 or Z_mat[i, j] < 0.7 else "#0F172A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.1, fontweight="bold", color=tcol)

    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels(metric_lbls, fontsize=8.2)
    ax.set_yticks(np.arange(len(mods)))
    ax.set_yticklabels(row_lbls, fontsize=8.0)
    ax.set_title("B. Compétition générative (500 simulations, N=87) : écart Z vs Dorabella")
    cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("|Z| (écart en σ, plafonné à 6σ)", fontsize=8.2)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_generative_and_loocv.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved {out}")


def main():
    with open(os.path.join(DATA_DIR, "transcriptions.json")) as f:
        tdata = json.load(f)
    with open(os.path.join(DATA_DIR, "results_01_geometry.json")) as f:
        r1 = json.load(f)
    with open(os.path.join(DATA_DIR, "results_02_corpus.json")) as f:
        r2 = json.load(f)
    with open(os.path.join(DATA_DIR, "results_03_generative.json")) as f:
        r3 = json.load(f)

    print("[04] Generating 4 publication-grade figures (300 DPI)...")
    make_fig1(tdata, r1)
    make_fig2(r1)
    make_fig3(r2)
    make_fig4(r3)
    print("[04] All figures generated successfully.")


if __name__ == "__main__":
    main()
