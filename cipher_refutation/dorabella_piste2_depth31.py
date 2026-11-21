import math
import random
from collections import Counter
import numpy as np
import sys
sys.path.insert(0, '/home/user')
from dorabella_piste1_test import LangModel, CORPUS_EN, CORPUS_IT, CORPUS_DE, clean_text

LINES = [
    "A2 E3 B2 A3 A1 C2 G1 A3 D1 H2 B3 F2 F1 B1 F2 C3 F2 F2 C2 E3 E3 F2 B1 H1 H2 H1 C1 B3 F3".split(),
    "G1 F2 G1 C2 H1 A3 D1 D2 A3 B2 F2 F2 B2 C2 C1 F1 G1 F2 B3 F2 C2 G2 F3 F1 B1 H1 D1 D1 H1 B3 F3".split(),
    "B2 F3 C2 G2 F3 B2 B1 G2 G3 C1 F3 B2 F2 C2 G2 F1 F3 C1 A3 E3 C1 F3 C2 A3 B1 H1 A3".split()
]
TOKENS = [t for l in LINES for t in l]
ORI = [ord(t[0]) - 65 for t in TOKENS]
ARC = [int(t[1]) - 1 for t in TOKENS]

ALPH24 = "ABCDEFGHIKLMNOPQRSTUWXYZ" # I/J, U/V merged
A24_TO_26 = [ord(c) - 65 for c in ALPH24]

def to24_ids(txt):
    t = clean_text(txt).replace("J", "I").replace("V", "U")
    return [ALPH24.index(c) for c in t]

# ==============================================================================
# 1. EXACT BIGRAM-VITERBI + TRIGRAM COORDINATE ASCENT FOR 3-LINE DEPTH (L=31)
# ==============================================================================
def solve_depth_31(cnums, lm, mode="vig", a24_to_26=A24_TO_26):
    """
    Given 87 cipher numbers in 0..23 split into B1(0..30), B2(31..61), B3(62..86),
    find the optimal 31-element key K[0..30] in 0..23 maximizing joint trigram score
    of P1, P2, P3 under mode ('vig': c-k, 'beau': k-c, 'vbeau': c+k, or 'nih': mod8 x mod3).
    """
    blocks = [cnums[0:31], cnums[31:62], cnums[62:87]]
    lp1, lp2, lp3 = lm.logp1, lm.logp2, lm.logp3

    def dec_val(c, k):
        if mode == "vig":
            return a24_to_26[(c - k) % 24]
        elif mode == "beau":
            return a24_to_26[(k - c) % 24]
        elif mode == "vbeau":
            return a24_to_26[(c + k) % 24]
        elif mode == "nih":
            co, ca = c // 3, c % 3
            ko, ka = k // 3, k % 3
            return a24_to_26[((co - ko) % 8) * 3 + ((ca - ka) % 3)]

    # Precompute decoded letter (0..25) for each (block, col, k)
    dec_tbl = []
    for b in blocks:
        tbl = [[dec_val(b[j], k) for k in range(24)] for j in range(len(b))]
        dec_tbl.append(tbl)

    # Step 1: Exact Bigram Viterbi over K[0..30]
    dp = np.full((31, 24), -1e18)
    back = np.zeros((31, 24), dtype=np.int32)
    for k in range(24):
        s = 0.0
        for bi, b in enumerate(blocks):
            s += lp1[dec_tbl[bi][0][k]]
        dp[0, k] = s

    for j in range(1, 31):
        active_b = [bi for bi, b in enumerate(blocks) if j < len(b)]
        for k in range(24):
            best_v = -1e18
            best_p = 0
            # Transition from prev key pk
            for pk in range(24):
                sc = dp[j - 1, pk]
                for bi in active_b:
                    sc += lp2[dec_tbl[bi][j - 1][pk], dec_tbl[bi][j][k]]
                if sc > best_v:
                    best_v = sc
                    best_p = pk
            dp[j, k] = best_v
            back[j, k] = best_p

    # Backtrack Viterbi key
    key = [0] * 31
    key[30] = int(np.argmax(dp[30]))
    for j in range(29, -1, -1):
        key[j] = int(back[j + 1, key[j + 1]])

    # Step 2: Trigram Coordinate Ascent + Adjacent-Pair Polish
    def score_all(kvec):
        tot = 0.0
        for bi, b in enumerate(blocks):
            L = len(b)
            p = [dec_tbl[bi][j][kvec[j]] for j in range(L)]
            tot += lp1[p[0]] + lp2[p[0], p[1]]
            for j in range(L - 2):
                tot += lp3[p[j], p[j + 1], p[j + 2]]
        return tot

    cur = score_all(key)
    for _ in range(8):
        improved = False
        for j in range(31):
            best_k, best_s = key[j], cur
            for kv in range(24):
                if kv == key[j]:
                    continue
                key[j] = kv
                sc = score_all(key)
                if sc > best_s + 1e-9:
                    best_s, best_k = sc, kv
                    improved = True
            key[j] = best_k
            cur = best_s
        if not improved:
            break

    pts = ["".join(chr(65 + dec_tbl[bi][j][key[j]]) for j in range(len(b))) for bi, b in enumerate(blocks)]
    key_str = "".join(chr(65 + a24_to_26[k]) for k in key)
    return cur, key_str, pts

# ==============================================================================
# 2. CALIBRATION: POSITIVE CONTROL (3 REAL ENGLISH LINES + RANDOM 31-CHAR OTP)
#                 vs NEGATIVE CONTROL (SHUFFLED DORABELLA)
# ==============================================================================
lm_en = LangModel("English", CORPUS_EN)
lm_it = LangModel("Italiano", CORPUS_IT)
lm_de = LangModel("Deutsch", CORPUS_DE)

print("=" * 76)
print("ETAPE 1 : CALIBRATION DU SOLVEUR EN PROFONDEUR (VITERBI + TRIGRAMMES)")
print("=" * 76)

# Positive control: 87 chars of real unseen English encrypted with a COMPLETELY RANDOM 31-letter key!
pt_ctrl_raw = (
    "DEAR MISS PENNY WE ENJOYED OUR "
    "VISIT TO WOLVERHAMPTON SO MUCH "
    "AND HOPE TO SEE YOU AT MALVERN VERY SOON INDEED"
)
pt_ctrl_ids = to24_ids(pt_ctrl_raw)[:87]
pt_ctrl_str = "".join(ALPH24[x] for x in pt_ctrl_ids)
rng = random.Random(1897)
true_key = [rng.randrange(24) for _ in range(31)]
ct_ctrl = [(pt_ctrl_ids[i] + true_key[i % 31]) % 24 for i in range(87)]

sc_pos, k_pos, pts_pos = solve_depth_31(ct_ctrl, lm_en, mode="vig")
rec_str = "".join(pts_pos)
acc_pos = sum(1 for a, b in zip(pt_ctrl_str, rec_str) if a == b) / 87.0
print(f"[CONTROLE POSITIF — 3 lignes anglaises + cle ALEATOIRE de 31 lettres]")
print(f"  Vrai B1 : {pt_ctrl_str[0:31]}")
print(f"  Rec  B1 : {pts_pos[0]}")
print(f"  Vrai B2 : {pt_ctrl_str[31:62]}")
print(f"  Rec  B2 : {pts_pos[1]}")
print(f"  Vrai B3 : {pt_ctrl_str[62:87]}")
print(f"  Rec  B3 : {pts_pos[2]}")
print(f"  Score = {sc_pos:.1f} | Exactitude recuperee = {100*acc_pos:.1f}% (sans connaitre la cle !)")

# Negative control: 5 completely random 87-symbol sequences
neg_scores = []
for s in range(5):
    rc = [rng.randrange(24) for _ in range(87)]
    sc_n, _, _ = solve_depth_31(rc, lm_en, mode="vig")
    neg_scores.append(sc_n)
print(f"\n[CONTROLE NEGATIF — 5 textes aleatoires 87 car., meme solveur 3x31]")
print(f"  Scores : moy = {sum(neg_scores)/len(neg_scores):.1f}, max = {max(neg_scores):.1f}")

# ==============================================================================
# 3. FULL 3x31 DEPTH ATTACK ON DORABELLA ACROSS ALL 64 GRIDS x 4 MODES
# ==============================================================================
print("\n" + "=" * 76)
print("ETAPE 2 : ATTAQUE EN PROFONDEUR 3x31 SUR DORABELLA (64 GRILLES x 4 MODES)")
print("=" * 76)

grids = []
for start in range(8):
    for direction in (1, -1):
        m_row = {( (start + direction * step) % 8, a ): a * 8 + step for a in range(3) for step in range(8)}
        grids.append((f"row_s{start}_d{direction}", m_row))
        m_col = {( (start + direction * step) % 8, a ): step * 3 + a for a in range(3) for step in range(8)}
        grids.append((f"col_s{start}_d{direction}", m_col))

top_results = []
for gname, gmap in grids:
    cnums = [gmap[(ORI[i], ARC[i])] for i in range(87)]
    for mode in ("vig", "beau", "vbeau", "nih"):
        sc, kstr, pts = solve_depth_31(cnums, lm_en, mode=mode)
        if len(top_results) < 8 or sc > top_results[-1][0]:
            top_results.append((sc, f"{mode:5s} {gname}", kstr, pts))
            top_results.sort(key=lambda x: -x[0])
            top_results = top_results[:8]

print(f"Total grilles x modes resolus par Viterbi+Trigrammes : {len(grids)*4}")
print(f"Reference : Controle positif = {sc_pos:.1f} | Hasard moy = {sum(neg_scores)/len(neg_scores):.1f} (max = {max(neg_scores):.1f})\n")
for rank, (sc, desc, kstr, pts) in enumerate(top_results[:5], 1):
    print(f"#{rank}  score={sc:.1f} | {desc} | cle={kstr}")
    print(f"    B1: {pts[0]}")
    print(f"    B2: {pts[1]}")
    print(f"    B3: {pts[2]}")

# Also test IT and DE on top 8 grids (row/col s0..s3 d1) to check multilingual depth!
print("\n" + "=" * 76)
print("ETAPE 3 : VERIFICATION EN PROFONDEUR 3x31 EN ITALIEN ET EN ALLEMAND")
print("=" * 76)
for lname, lm in [("IT", lm_it), ("DE", lm_de)]:
    # Negative baseline
    neg_l = [solve_depth_31([rng.randrange(24) for _ in range(87)], lm, mode="vig")[0] for _ in range(3)]
    best_l = (-1e18, "", "", [])
    for gname, gmap in grids[:32]: # all row/col clockwise & counter-clockwise
        cnums = [gmap[(ORI[i], ARC[i])] for i in range(87)]
        for mode in ("vig", "beau"):
            sc, kstr, pts = solve_depth_31(cnums, lm, mode=mode)
            if sc > best_l[0]:
                best_l = (sc, f"{mode} {gname}", kstr, pts)
    print(f"[{lname}] Hasard moy={sum(neg_l)/len(neg_l):.1f} (max={max(neg_l):.1f}) | Meilleur Dorabella={best_l[0]:.1f} ({best_l[1]})")
    print(f"    B1: {best_l[3][0]}")
    print(f"    B2: {best_l[3][1]}")
    print(f"    B3: {best_l[3][2]}")

# ==============================================================================
# 4. CRIB-DRAGGING CIBLE SUR B1/B2/B3 (MOTS PROBABLES D'ELGAR)
# ==============================================================================
print("\n" + "=" * 76)
print("ETAPE 4 : CRIB-DRAGGING CIBLE (DORABELLA, MISSPENNY, EDWARDELGAR, THANKYOU...)")
print("=" * 76)

CRIBS = ["DORABELLA", "MISSPENNY", "EDWARDELGAR", "WOLUERHAMPTON", "MALUERN", "THANKYOU", "DEARDORA", "DEARPENNY", "ENIGMA", "IULYFOURTEEN", "COMETO", "WITHLOUE"]
# Test cribs on B1, B2, B3 across all 32 standard grids (vig & beau) and check if the OTHER blocks produce high English trigram scores!
best_cribs = []
for crib in CRIBS:
    c_ids = [ALPH24.index(ch) for ch in crib]
    clen = len(c_ids)
    for gname, gmap in grids:
        cnums = [gmap[(ORI[i], ARC[i])] for i in range(87)]
        blks = [cnums[0:31], cnums[31:62], cnums[62:87]]
        for mode in ("vig", "beau"):
            for target_b in range(3):
                tb = blks[target_b]
                for pos in range(len(tb) - clen + 1):
                    # Deduce key segment
                    if mode == "vig":
                        kseg = [(tb[pos + i] - c_ids[i]) % 24 for i in range(clen)]
                    else:
                        kseg = [(tb[pos + i] + c_ids[i]) % 24 for i in range(clen)]
                    # Decrypt corresponding slices in OTHER blocks
                    other_texts = []
                    sc_sum = 0.0
                    n_chars = 0
                    for ob_i, ob in enumerate(blks):
                        if ob_i == target_b:
                            continue
                        if pos + clen <= len(ob):
                            if mode == "vig":
                                dec = [ALPH24[(ob[pos + i] - kseg[i]) % 24] for i in range(clen)]
                            else:
                                dec = [ALPH24[(kseg[i] - ob[pos + i]) % 24] for i in range(clen)]
                            s = "".join(dec)
                            other_texts.append(f"B{ob_i+1}:{s}")
                            sc_sum += lm_en.score_text(s)
                            n_chars += clen
                    if n_chars > 0:
                        avg_sc = sc_sum / n_chars # per-char log-prob (good English ~ -2.3 to -2.6)
                        if len(best_cribs) < 8 or avg_sc > best_cribs[-1][0]:
                            kstr = "".join(ALPH24[k] for k in kseg)
                            best_cribs.append((avg_sc, crib, f"B{target_b+1}@{pos}", f"{mode} {gname}", kstr, " | ".join(other_texts)))
                            best_cribs.sort(key=lambda x: -x[0])
                            best_cribs = best_cribs[:8]

print("Meilleurs crib-drags (score moyen par lettre sur les AUTRES blocs ; bon anglais ~ -2.2 a -2.5):")
for avg_sc, crib, where, desc, kstr, others in best_cribs:
    print(f"  {avg_sc:5.2f}/ch | crib={crib:13s} ({where:5s}) | {desc:16s} | key={kstr:13s} -> {others}")

# ==============================================================================
# 5. FREE-ALPHABET STRUCTURAL CHECK: WHAT DOES THE 3x31 MATRIX ACTUALLY CONSTRAIN?
# ==============================================================================
print("\n" + "=" * 76)
print("ETAPE 5 : ANALYSE SANS GRILLE DES 13 COINCIDENCES DE LA MATRICE 3x31")
print("=" * 76)
b1, b2, b3 = TOKENS[0:31], TOKENS[31:62], TOKENS[62:87]
# Check if the 13 coincidences happen mainly in ONE zone (cols 17..28) or uniformly across 0..30!
early_m = sum(1 for i in range(17) for a, b in [(b1[i], b2[i])] if a == b)
late_m  = sum(1 for i in range(17, 31) for a, b in [(b1[i], b2[i])] if a == b)
print(f"  B1 vs B2 cols  0..16 : {early_m}/17 ({100*early_m/17:.1f}%)")
print(f"  B1 vs B2 cols 17..30 : {late_m}/14 ({100*late_m/14:.1f}%)  <-- CONCENTRATION MASSIVE EN FIN DE BLOC !")
print(f"  B1[17..28] : {' '.join(b1[17:29])}")
print(f"  B2[17..28] : {' '.join(b2[17:29])}")
print(f"  B3[17..24] : {' '.join(b3[17:25])}")
