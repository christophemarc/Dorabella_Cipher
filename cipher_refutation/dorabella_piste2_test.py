import math
import random
from collections import Counter
import numpy as np
from scipy.stats import binom
import sys
sys.path.insert(0, '/home/user')
from dorabella_piste1_test import LangModel, CORPUS_EN, CONTROLS, clean_text, solve_masc

LINES = [
    "A2 E3 B2 A3 A1 C2 G1 A3 D1 H2 B3 F2 F1 B1 F2 C3 F2 F2 C2 E3 E3 F2 B1 H1 H2 H1 C1 B3 F3".split(),
    "G1 F2 G1 C2 H1 A3 D1 D2 A3 B2 F2 F2 B2 C2 C1 F1 G1 F2 B3 F2 C2 G2 F3 F1 B1 H1 D1 D1 H1 B3 F3".split(),
    "B2 F3 C2 G2 F3 B2 B1 G2 G3 C1 F3 B2 F2 C2 G2 F1 F3 C1 A3 E3 C1 F3 C2 A3 B1 H1 A3".split()
]
TOKENS = [t for l in LINES for t in l]
N = len(TOKENS)
ORI = [ord(t[0]) - 65 for t in TOKENS]   # 0..7
ARC = [int(t[1]) - 1 for t in TOKENS]    # 0..2

def ic_of(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    c = Counter(seq)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))

# ==============================================================================
# PART A: DISSECTING LAG=31 (ORIENTATION vs ARC vs FULL SYMBOL)
# ==============================================================================
print("=" * 72)
print("PART A : AUTOCORRELATION SEPAREE (SYMBOLE / ORIENTATION / BOUCLES)")
print("=" * 72)
ic_sym = ic_of(TOKENS)
ic_ori = ic_of(ORI)
ic_arc = ic_of(ARC)
print(f"IC globaux : Symbole={ic_sym:.4f}, Orientation={ic_ori:.4f}, Boucles={ic_arc:.4f}")

for label, seq, base_p in [("SYMBOLE", TOKENS, ic_sym), ("ORIENTATION", ORI, ic_ori), ("BOUCLES", ARC, ic_arc)]:
    rows = []
    for lag in range(1, 46):
        m = sum(1 for i in range(N - lag) if seq[i] == seq[i + lag])
        tot = N - lag
        p = binom.sf(m - 1, tot, base_p)
        rows.append((p, lag, m, tot, m / tot))
    rows.sort()
    top3 = ", ".join(f"lag={lg} ({m}/{tot}={100*r:.1f}%, p={p:.4f})" for p, lg, m, tot, r in rows[:4])
    print(f"  {label:11s} top lags: {top3}")

d_ori_31 = Counter((ORI[i+31] - ORI[i]) % 8 for i in range(N - 31))
d_arc_31 = Counter((ARC[i+31] - ARC[i]) % 3 for i in range(N - 31))
print(f"\n  A lag=31, differences (ori[i+31]-ori[i]) mod 8 : {sorted(d_ori_31.items())}")
print(f"  A lag=31, differences (arc[i+31]-arc[i]) mod 3 : {sorted(d_arc_31.items())}")

print("\n=== TEST DE FRIEDMAN (IC MOYEN PAR TRANCHE POUR PERIODES p = 1..15) ===")
for p in range(1, 16):
    ics = [ic_of(TOKENS[o::p]) for o in range(p)]
    ic_o = [ic_of(ORI[o::p]) for o in range(p)]
    ic_a = [ic_of(ARC[o::p]) for o in range(p)]
    print(f"  p={p:2d} | IC_sym={sum(ics)/p:.4f} | IC_ori={sum(ic_o)/p:.4f} | IC_arc={sum(ic_a)/p:.4f}")

# ==============================================================================
# PART B: NIHILIST / 3x8 COORDINATE SHIFT ATTACK (MOD 8 & MOD 3)
# ==============================================================================
print("\n" + "=" * 72)
print("PART B : ATTAQUE NIHILISTE / COORDONNEES 3x8 (DECALAGE MOD 8 x MOD 3)")
print("=" * 72)

lm_en = LangModel("English", CORPUS_EN)

def optimize_coord_key(ori_seq, arc_seq, period, restarts=15, seed=42):
    rng = random.Random(seed)
    n = len(ori_seq)
    best_ic = -1.0
    best_key = None
    for _ in range(restarts):
        ko = [0] + [rng.randrange(8) for _ in range(period - 1)]
        ka = [0] + [rng.randrange(3) for _ in range(period - 1)]
        
        def eval_key():
            dec = [((ori_seq[i] - ko[i % period]) % 8) * 3 + ((arc_seq[i] - ka[i % period]) % 3) for i in range(n)]
            return ic_of(dec)
        
        cur = eval_key()
        improved = True
        while improved:
            improved = False
            for pos in range(1, period):
                best_o, best_a, best_v = ko[pos], ka[pos], cur
                for o in range(8):
                    for a in range(3):
                        ko[pos], ka[pos] = o, a
                        v = eval_key()
                        if v > best_v + 1e-9:
                            best_v, best_o, best_a = v, o, a
                            improved = True
                ko[pos], ka[pos] = best_o, best_a
                cur = best_v
        if cur > best_ic:
            best_ic = cur
            best_key = (ko[:], ka[:])
    dec_toks = [f"{(ori_seq[i]-best_key[0][i%period])%8}_{(arc_seq[i]-best_key[1][i%period])%3}" for i in range(n)]
    return best_ic, best_key, dec_toks

ALPH24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
def to24(txt):
    return txt.replace("J", "I").replace("V", "U")

pt87 = to24(CONTROLS["EN"][:87])
c24_ori = [ALPH24.index(c) % 8 for c in pt87]
c24_arc = [ALPH24.index(c) // 8 for c in pt87]
key_o_ctrl = [3, 6, 1, 5]
key_a_ctrl = [0, 2, 1, 2]
ct_o_ctrl = [(c24_ori[i] + key_o_ctrl[i % 4]) % 8 for i in range(87)]
ct_a_ctrl = [(c24_arc[i] + key_a_ctrl[i % 4]) % 3 for i in range(87)]
ct_sym_ctrl = [f"{o}_{a}" for o, a in zip(ct_o_ctrl, ct_a_ctrl)]

print(f"[CONTROLE POSITIF Nihiliste 3x8, p=4] IC avant attaque = {ic_of(ct_sym_ctrl):.4f}")
ic_rec, k_rec, dec_toks_ctrl = optimize_coord_key(ct_o_ctrl, ct_a_ctrl, 4, restarts=15, seed=1)
sc_ctrl, pt_rec = solve_masc(dec_toks_ctrl, lm_en, restarts=10, steps=5500, seed=1)
acc_ctrl = sum(1 for a, b in zip(pt87, pt_rec) if a == b) / 87.0
print(f"  IC apres alignement p=4 = {ic_rec:.4f} | score MASC = {sc_ctrl:.1f} | exactitude = {100*acc_ctrl:.1f}%")
print(f"  Recupere : {pt_rec}")

rng_sh = random.Random(2026)
sh_idx = list(range(N))
rng_sh.shuffle(sh_idx)
sh_ori = [ORI[i] for i in sh_idx]
sh_arc = [ARC[i] for i in sh_idx]

print("\n[DORABELLA vs HASARD] Alignement Nihiliste 3x8 pour periodes p = 2..8 :")
for p in range(2, 9):
    ic_d, key_d, dec_d = optimize_coord_key(ORI, ARC, p, restarts=15, seed=10+p)
    ic_r, _, _ = optimize_coord_key(sh_ori, sh_arc, p, restarts=10, seed=50+p)
    sc_d, txt_d = solve_masc(dec_d, lm_en, restarts=10, steps=5000, seed=20+p)
    print(f"  p={p} | Dorabella IC={ic_d:.4f} (Hasard IC={ic_r:.4f}) | MASC={sc_d:6.1f} | {txt_d[:52]}...")

# ==============================================================================
# PART C: VIGENERE / BEAUFORT / AUTOKEY / KEY-RESET-PER-LINE ON 24-LETTER GRIDS
# ==============================================================================
print("\n" + "=" * 72)
print("PART C : VIGENERE / BEAUFORT / AUTOKEY / CLE PAR LIGNE (MOTS-CLES ELGAR)")
print("=" * 72)

KEYWORDS = [
    "DORA", "DORABELLA", "PENNY", "MISSPENNY", "ELGAR", "EDWARD", "ALICE", "EDOO", "EDU",
    "ENIGMA", "MALVERN", "FORLI", "WOLVERHAMPTON", "WORCESTER", "JULY", "FOURTEEN",
    "MOZART", "COSI", "COSIFANTUTTE", "MUSIC", "CIPHER", "CRYPTOGRAM", "NIHILIST",
    "BEACON", "BACH", "SCHUMANN", "BRAHMS", "LUTHER", "CHERUBINI", "VARIATION",
    "SECRET", "FRIEND", "LOVE", "CARICE", "TROYTE", "NIMROD", "WAND", "YOUTH",
    "SUMMER", "GARDEN", "VIOLIN", "ORGAN", "RECTORY", "ENGLAND", "A", "E", "EE",
    "SCHOOLING", "PALLMALL", "POLYBIUS", "ALPHA", "BETA", "LARKS", "CLOAK"
]

def make_grid_mappings():
    grids = []
    for start in range(8):
        for direction in (1, -1):
            m = {}
            for a in range(3):
                for step in range(8):
                    o = (start + direction * step) % 8
                    m[(o, a)] = a * 8 + step
            grids.append((f"row_s{start}_d{direction}", m))
    for start in range(8):
        for direction in (1, -1):
            m = {}
            for step in range(8):
                o = (start + direction * step) % 8
                for a in range(3):
                    m[(o, a)] = step * 3 + a
            grids.append((f"col_s{start}_d{direction}", m))
    return grids

grids = make_grid_mappings()
ALPH24_VARIANTS = [
    ("IJ_UV", "ABCDEFGHIKLMNOPQRSTUWXYZ"),
    ("IJ_VW", "ABCDEFGHIKLMNOPQRSTUVXYZ"),
]

top_hits = []
for aname, a24 in ALPH24_VARIANTS:
    ch2idx = {c: i for i, c in enumerate(a24)}
    for gname, gmap in grids:
        cnums = [gmap[(ORI[i], ARC[i])] for i in range(N)]
        
        for mode in ("autokey_pt", "autokey_ct"):
            for seed_k in range(24):
                if mode == "autokey_pt":
                    p = []
                    k = seed_k
                    for c in cnums:
                        pv = (c - k) % 24
                        p.append(pv)
                        k = pv
                else:
                    p = []
                    k = seed_k
                    for c in cnums:
                        p.append((c - k) % 24)
                        k = c
                txt = "".join(a24[x] for x in p)
                sc = lm_en.score_text(txt)
                if len(top_hits) < 8 or sc > top_hits[-1][0]:
                    top_hits.append((sc, f"{mode}(k={a24[seed_k]}) {aname} {gname}", txt))
                    top_hits.sort(key=lambda x: -x[0])
                    top_hits = top_hits[:8]

        for kw in KEYWORDS:
            kw_clean = to24(kw) if aname == "IJ_UV" else kw.replace("J", "I").replace("W", "V")
            knums = [ch2idx[c] for c in kw_clean if c in ch2idx]
            if not knums:
                continue
            kl = len(knums)
            for reset_line in (False, True):
                for vmode in ("vig", "beau", "vbeau"):
                    if not reset_line:
                        if vmode == "vig":
                            p = [(cnums[i] - knums[i % kl]) % 24 for i in range(N)]
                        elif vmode == "beau":
                            p = [(knums[i % kl] - cnums[i]) % 24 for i in range(N)]
                        else:
                            p = [(cnums[i] + knums[i % kl]) % 24 for i in range(N)]
                    else:
                        p = []
                        idx = 0
                        for L in LINES:
                            for j in range(len(L)):
                                c = cnums[idx]
                                kv = knums[j % kl]
                                if vmode == "vig": p.append((c - kv) % 24)
                                elif vmode == "beau": p.append((kv - c) % 24)
                                else: p.append((c + kv) % 24)
                                idx += 1
                    txt = "".join(a24[x] for x in p)
                    sc = lm_en.score_text(txt)
                    if len(top_hits) < 8 or sc > top_hits[-1][0]:
                        rl = "_perLine" if reset_line else ""
                        top_hits.append((sc, f"{vmode}{rl}({kw}) {aname} {gname}", txt))
                        top_hits.sort(key=lambda x: -x[0])
                        top_hits = top_hits[:8]

print(f"Total combinaisons testees (Part C) : {len(ALPH24_VARIANTS)*len(grids)*(48 + len(KEYWORDS)*6):,}")
print("Meilleurs scores (rappel : vrai anglais 87 car. = -230.0 a -214.9):")
for sc, desc, txt in top_hits:
    print(f"  score={sc:6.1f} | {desc:45s} | {txt[:50]}...")

# ==============================================================================
# PART D: TEST DU CHIFFREMENT EN PROFONDEUR (3 LIGNES = MEME CLE)
# ==============================================================================
print("\n" + "=" * 72)
print("PART D : 3 LIGNES COMPAREES PAR DECALAGE RELATIF (-5..+5)")
print("=" * 72)
for name, la, lb in [("L1 vs L2", LINES[0], LINES[1]), ("L2 vs L3", LINES[1], LINES[2]), ("L1 vs L3", LINES[0], LINES[2])]:
    best_sh = []
    for sh in range(-5, 6):
        m = 0
        tot = 0
        for i in range(len(la)):
            j = i + sh
            if 0 <= j < len(lb):
                tot += 1
                if la[i] == lb[j]:
                    m += 1
        p = binom.sf(m - 1, tot, ic_sym)
        best_sh.append((p, sh, m, tot))
    best_sh.sort()
    s_str = ", ".join(f"shift={sh:+d}: {m}/{tot} ({100*m/tot:.1f}%, p={p:.4f})" for p, sh, m, tot in best_sh[:3])
    print(f"  {name}: {s_str}")
