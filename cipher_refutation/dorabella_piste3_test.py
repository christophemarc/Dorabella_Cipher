import math
import random
from collections import Counter
import numpy as np
from scipy.stats import chi2_contingency
import sys
sys.path.insert(0, '/home/user')
from dorabella_piste1_test import LangModel, CORPUS_EN, CORPUS_IT, CORPUS_DE, CONTROLS, clean_text, solve_masc

LINES = [
    "A2 E3 B2 A3 A1 C2 G1 A3 D1 H2 B3 F2 F1 B1 F2 C3 F2 F2 C2 E3 E3 F2 B1 H1 H2 H1 C1 B3 F3".split(),
    "G1 F2 G1 C2 H1 A3 D1 D2 A3 B2 F2 F2 B2 C2 C1 F1 G1 F2 B3 F2 C2 G2 F3 F1 B1 H1 D1 D1 H1 B3 F3".split(),
    "B2 F3 C2 G2 F3 B2 B1 G2 G3 C1 F3 B2 F2 C2 G2 F1 F3 C1 A3 E3 C1 F3 C2 A3 B1 H1 A3".split()
]
TOKENS = [t for l in LINES for t in l]
N = len(TOKENS)
ORI = [ord(t[0]) - 65 for t in TOKENS]
ARC = [int(t[1]) - 1 for t in TOKENS]

def ic_of(seq):
    n = len(seq)
    c = Counter(seq)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))

lm_en = LangModel("English", CORPUS_EN)
lm_it = LangModel("Italiano", CORPUS_IT)
lm_de = LangModel("Deutsch", CORPUS_DE)

# ==============================================================================
# PART A : FRACTIONNEMENT BIFIDE 3x8 (DECALAGE / BLOCS ENTRE ORIENTATION & BOUCLES)
# ==============================================================================
print("=" * 76)
print("PART A : TESTS DE FRACTIONNEMENT (3x8 ORIENTATION/BOUCLES & BIFIDE 5x5)")
print("=" * 76)

# 1. Test shifting ARC relative to ORI by lag s in -15..+15
print("1. Recombinaison decalee (ori[i], arc[i + s]) pour s = -15..+15 :")
shift_stats = []
for s in range(-15, 16):
    recomb = [(ORI[i], ARC[(i + s) % N]) for i in range(N)]
    ic_v = ic_of(recomb)
    # Chi2 independence
    tbl = [[sum(1 for o, a in recomb if o == oi and a == ai) for ai in range(3)] for oi in range(8)]
    chi2, p, _, _ = chi2_contingency(tbl)
    shift_stats.append((ic_v, chi2, p, s, [f"{o}_{a}" for o, a in recomb]))

shift_stats.sort(key=lambda x: -x[0])
for ic_v, chi2, p, s, toks in shift_stats[:6]:
    tag = "  <-- ORIGINAL (s=0)" if s == 0 else ""
    print(f"   s={s:+3d} | IC={ic_v:.4f} | chi2={chi2:5.2f} (p={p:.4f}){tag}")

# Run MASC on top 2 non-zero shifts just to be thorough
print("   Test MASC sur les 2 meilleurs decalages non nuls :")
non_zero = [x for x in shift_stats if x[3] != 0][:2]
for ic_v, chi2, p, s, toks in non_zero:
    sc, dec = solve_masc(toks, lm_en, restarts=8, steps=4500, seed=300+s)
    print(f"   s={s:+3d} -> MASC EN = {sc:.1f} | {dec[:55]}...")

# 2. Test block-level fractionation of ARC (reverse / shift inside blocks of size p=2..15)
print("\n2. Recombinaison par blocs (inversion ou rotation des boucles par bloc p=2..15) :")
best_blk = []
for p in range(2, 16):
    # Reverse arcs inside each block of size p
    arc_rev = []
    for i in range(0, N, p):
        arc_rev.extend(reversed(ARC[i:i+p]))
    recomb = [f"{ORI[i]}_{arc_rev[i]}" for i in range(N)]
    best_blk.append((ic_of(recomb), f"rev_arc_p{p}", recomb))
    
    # Rotate arcs by half-block inside each block
    arc_rot = []
    h = max(1, p // 2)
    for i in range(0, N, p):
        blk = ARC[i:i+p]
        arc_rot.extend(blk[h:] + blk[:h])
    recomb2 = [f"{ORI[i]}_{arc_rot[i]}" for i in range(N)]
    best_blk.append((ic_of(recomb2), f"rot_arc_p{p}", recomb2))

best_blk.sort(key=lambda x: -x[0])
for ic_v, name, toks in best_blk[:5]:
    sc, dec = solve_masc(toks, lm_en, restarts=6, steps=4000, seed=400)
    print(f"   {name:14s} | IC={ic_v:.4f} | MASC EN={sc:6.1f} | {dec[:50]}...")

# 3. Benchmark: what IC does a REAL 5x5 Bifid cipher produce on 87-char English?
print("\n3. Profil IC d'un vrai chiffre Bifide 5x5 sur 87 car. d'anglais (periodes 5, 7, 9, 87) :")
SQ25 = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
pt_en87 = CONTROLS["EN"][:87].replace("J", "I")
def bifid5_encrypt(txt, period):
    out = []
    for i in range(0, len(txt), period):
        blk = txt[i:i+period]
        rows = [SQ25.index(c) // 5 for c in blk]
        cols = [SQ25.index(c) % 5 for c in blk]
        comb = rows + cols
        for k in range(0, len(comb), 2):
            out.append(SQ25[comb[k] * 5 + comb[k + 1]])
    return out

for per in (5, 7, 9, 15, 87):
    ct_b = bifid5_encrypt(pt_en87, per)
    print(f"   Bifide 5x5 (p={per:2d}) : IC={ic_of(ct_b):.4f}, symboles uniques={len(set(ct_b))}/25  (Dorabella : IC=0.0585, uniques=20/24)")

# ==============================================================================
# PART B : SOLVEUR HOMOPHONIQUE CALIBRE (AVEC CONTROLE ANTI-SURAPPRENTISSAGE)
# ==============================================================================
print("\n" + "=" * 76)
print("PART B : ATTAQUE HOMOPHONIQUE (2 SYMBOLES PAR LETTRE FREQUENTE)")
print("=" * 76)

# Target English unigram probabilities (from lm_en) to penalize degenerate EEEEE/TTTTT collapses
def solve_homophonic(cipher_tokens, lm, max_merges=3, max_per_letter=2, restarts=16, steps=7000, seed=1897):
    """
    Maps each of the n_uniq cipher symbols to a plaintext letter 0..25 such that:
    - at most `max_per_letter` symbols share the same plaintext letter (default 2),
    - at most `max_merges` total homophonic collisions occur (so n_distinct_letters >= n_uniq - max_merges).
    Includes a soft KL-divergence penalty on letter frequencies to prevent collapsing onto E/T/A.
    """
    rng = random.Random(seed)
    counts = Counter(cipher_tokens)
    uniq = [u for u, _ in counts.most_common()]
    sym2id = {u: i for i, u in enumerate(uniq)}
    c_ids = [sym2id[t] for t in cipher_tokens]
    n_uniq = len(uniq)
    n_len = len(c_ids)
    lp1, lp2, lp3 = lm.logp1, lm.logp2, lm.logp3
    target_p1 = np.exp(lp1)

    def eval_mapping(m):
        seq = [m[c] for c in c_ids]
        sc = lp1[seq[0]] + lp2[seq[0], seq[1]]
        for k in range(n_len - 2):
            sc += lp3[seq[k], seq[k + 1], seq[k + 2]]
        # Light unigram chi2/KL regularization so solver can't spam 30% E's
        obs = np.bincount(seq, minlength=26) / n_len
        kl = 0.0
        for a in range(26):
            if obs[a] > 0:
                kl += obs[a] * math.log(obs[a] / target_p1[a])
        # Weight = 25 * KL (very light for natural English ~0.12 -> -3 pts, heavy for spam ~0.6 -> -15 pts)
        return float(sc - 25.0 * kl), float(sc)

    best_obj = -1e18
    best_raw = -1e18
    best_m = None

    for r in range(restarts):
        # Start from injective assignment
        m = lm.freq_order[:n_uniq]
        if r > 0:
            pool = lm.freq_order[:]
            for _ in range(rng.randint(5, 20)):
                i, j = rng.randrange(26), rng.randrange(26)
                pool[i], pool[j] = pool[j], pool[i]
            m = pool[:n_uniq]

        cur_obj, cur_raw = eval_mapping(m)
        best_r_obj, best_r_raw, best_r_m = cur_obj, cur_raw, m[:]

        t0, t1 = 4.0, 0.06
        for step in range(steps):
            old_m = m[:]
            # Two move types: (1) swap two symbol assignments, (2) reassign one symbol to any letter (respecting constraints)
            if rng.random() < 0.45:
                i, j = rng.sample(range(n_uniq), 2)
                m[i], m[j] = m[j], m[i]
            else:
                i = rng.randrange(n_uniq)
                new_l = rng.randrange(26)
                if new_l == m[i]:
                    continue
                # Check constraints: max_per_letter & max_merges
                m[i] = new_l
                cnt = Counter(m)
                if cnt[new_l] > max_per_letter:
                    m = old_m
                    continue
                merges = n_uniq - len(cnt)
                if merges > max_merges:
                    m = old_m
                    continue

            new_obj, new_raw = eval_mapping(m)
            diff = new_obj - cur_obj
            if diff >= 0:
                cur_obj, cur_raw = new_obj, new_raw
                if cur_obj > best_r_obj:
                    best_r_obj, best_r_raw, best_r_m = cur_obj, cur_raw, m[:]
            else:
                temp = t0 * ((t1 / t0) ** (step / steps))
                if math.exp(diff / temp) > rng.random():
                    cur_obj, cur_raw = new_obj, new_raw
                else:
                    m = old_m

        # Polish: single-symbol moves & pairwise swaps
        m = best_r_m[:]
        cur_obj, cur_raw = best_r_obj, best_r_raw
        improved = True
        while improved:
            improved = False
            for i in range(n_uniq):
                orig = m[i]
                for new_l in range(26):
                    if new_l == orig:
                        continue
                    m[i] = new_l
                    cnt = Counter(m)
                    if cnt[new_l] > max_per_letter or (n_uniq - len(cnt)) > max_merges:
                        m[i] = orig
                        continue
                    no, nr = eval_mapping(m)
                    if no > cur_obj + 1e-9:
                        cur_obj, cur_raw = no, nr
                        orig = new_l
                        improved = True
                    else:
                        m[i] = orig
            for i in range(n_uniq):
                for j in range(i + 1, n_uniq):
                    if m[i] == m[j]:
                        continue
                    m[i], m[j] = m[j], m[i]
                    no, nr = eval_mapping(m)
                    if no > cur_obj + 1e-9:
                        cur_obj, cur_raw = no, nr
                        improved = True
                    else:
                        m[i], m[j] = m[j], m[i]

        if cur_obj > best_obj:
            best_obj, best_raw, best_m = cur_obj, cur_raw, m[:]

    pt = "".join(chr(65 + best_m[c]) for c in c_ids)
    # Identify merged homophones
    by_let = {}
    for sym, lid in zip(uniq, best_m):
        by_let.setdefault(chr(65 + lid), []).append(sym)
    homos = {k: v for k, v in by_let.items() if len(v) > 1}
    return best_obj, best_raw, pt, homos

# --- POSITIVE CONTROL: 87-char English encrypted with 3 homophones (E, T, A each split in 2) ---
rng = random.Random(999)
pt_h_ctrl = CONTROLS["EN"][:87]
# Assign homophones to E, T, A (so IC drops from ~0.066 to ~0.055-0.058, just like Dorabella!)
ct_h_ctrl = []
for ch in pt_h_ctrl:
    if ch in ("E", "T", "A"):
        ct_h_ctrl.append(f"{ch}_{rng.randrange(2)}")
    else:
        ct_h_ctrl.append(f"{ch}_0")

print(f"[CONTROLE POSITIF Homophonique EN (E,T,A dedoubles)]")
print(f"  IC = {ic_of(ct_h_ctrl):.4f}, symboles uniques = {len(set(ct_h_ctrl))}  (Dorabella : IC=0.0585, uniques=20)")
obj_p, raw_p, dec_p, hom_p = solve_homophonic(ct_h_ctrl, lm_en, max_merges=4, restarts=14, steps=6000, seed=42)
acc_p = sum(1 for a, b in zip(pt_h_ctrl, dec_p) if a == b) / 87.0
print(f"  Original : {pt_h_ctrl}")
print(f"  Solveur  : {dec_p}")
print(f"  Score obj={obj_p:.1f} (raw={raw_p:.1f}) | Exactitude={100*acc_p:.1f}% | Homophones trouves={hom_p}")

# --- NEGATIVE CONTROL: 3 random shuffles of Dorabella ---
neg_objs = []
for si in range(3):
    sh = TOKENS[:]
    rng.shuffle(sh)
    o_n, r_n, _, _ = solve_homophonic(sh, lm_en, max_merges=4, restarts=8, steps=4500, seed=10+si)
    neg_objs.append((o_n, r_n))
mean_neg_obj = sum(x[0] for x in neg_objs) / len(neg_objs)
max_neg_obj = max(x[0] for x in neg_objs)
print(f"\n[CONTROLE NEGATIF Homophonique (max_merges=4, Dorabella melange)]")
print(f"  Score obj moy = {mean_neg_obj:.1f} (max = {max_neg_obj:.1f})")

# --- ATTACK DORABELLA (T0 Normal & T2 Lignes inversees) FOR k = 1, 2, 3, 4, 5 merges ---
print("\n[DORABELLA — ATTAQUE HOMOPHONIQUE EN ANGLAIS (T0 Normal & T2 Lignes inversees)]")
t2_seq = list(reversed(LINES[0])) + list(reversed(LINES[1])) + list(reversed(LINES[2]))

for label, seq in [("T0_Normal", TOKENS), ("T2_Lignes_inv", t2_seq)]:
    for km in (2, 3, 4, 5):
        obj_d, raw_d, dec_d, hom_d = solve_homophonic(seq, lm_en, max_merges=km, restarts=14, steps=6000, seed=700+km)
        print(f"  {label:13s} (k={km} fusions) | obj={obj_d:6.1f} (raw={raw_d:6.1f}) | homos={hom_d}")
        print(f"     -> {dec_d}")

# --- QUICK CHECK HOMOPHONIQUE EN ITALIEN & ALLEMAND (k=4) ---
print("\n[DORABELLA — ATTAQUE HOMOPHONIQUE EN ITALIEN ET ALLEMAND (T0 Normal, k=4)]")
for lname, lm in [("IT", lm_it), ("DE", lm_de)]:
    # Quick negative baseline
    sh = TOKENS[:]
    rng.shuffle(sh)
    o_n, _, _, _ = solve_homophonic(sh, lm, max_merges=4, restarts=6, steps=4000, seed=88)
    o_d, r_d, dec_d, hom_d = solve_homophonic(TOKENS, lm, max_merges=4, restarts=12, steps=5500, seed=89)
    print(f"  [{lname}] Dorabella obj={o_d:.1f} (raw={r_d:.1f}) vs Hasard={o_n:.1f} | homos={hom_d}")
    print(f"     -> {dec_d}")
