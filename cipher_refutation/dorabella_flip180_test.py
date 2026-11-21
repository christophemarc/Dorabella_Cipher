import itertools
import math
import random
import time
from collections import Counter
import numpy as np

# Load clean helpers & corpora without executing top-level blocks
with open("/home/user/dorabella_phonetic_transcriptions_test.py") as f:
    _code = f.read().split("# MAIN BENCHMARK EXECUTION")[0]
_ns = {}
exec(_code, _ns)

HAUER = _ns["HAUER_2025"]
CORPUS_EN_FULL = _ns["CORPUS_EN_FULL"]
CORPUS_FR_FULL = _ns["CORPUS_FR_FULL"]
LangModel = _ns["LangModel"]
solve_masc = _ns["solve_masc"]
en_std = _ns["en_std"]
tokenize_words = _ns["tokenize_words"]
ic_of = _ns["ic_of"]

O = "ABCDEFGH"
o2i = {c: i for i, c in enumerate(O)}

def parse(s):
    return o2i[s[0]], int(s[1])

def fmt(o, a):
    return f"{O[o % 8]}{a}"

def rot180(s):
    o, a = parse(s)
    return fmt(o + 4, a)

def en_backslang(text):
    return "".join(w[::-1] for w in tokenize_words(text))

# Build clean language models (hold out last 1500 chars for positive controls)
EN_TR = CORPUS_EN_FULL[:-1500]
EN_TE = CORPUS_EN_FULL[-1500:]
FR_TR = CORPUS_FR_FULL[:-900]

lm_en = LangModel(en_std(EN_TR), "EN_STD")
lm_bk = LangModel(en_backslang(EN_TR), "EN_BACK")
lm_fr = LangModel(en_std(FR_TR), "FR_STD")

t0 = time.time()

# Positive & Negative controls for 87-char EN_STD
ctrl_en = en_std(EN_TE)[120:207]
sc_pos_en, dec_pos_en, _ = solve_masc(list(ctrl_en), lm_en, restarts=24, steps=8000, seed=10)
acc_pos_en = 100.0 * sum(a == b for a, b in zip(ctrl_en, dec_pos_en)) / 87.0
neg_scores = []
for k in range(5):
    sh = list(HAUER)
    random.Random(100 + k).shuffle(sh)
    sc, _, _ = solve_masc(sh, lm_en, restarts=14, steps=6000, seed=110 + k)
    neg_scores.append(sc)
neg_en = float(np.mean(neg_scores))

print("=" * 86)
print("CALIBRATION DE REFERENCE (EN_STD, 87 CARACTERES)")
print("=" * 86)
print(f"  PosCtrl EN_STD : {sc_pos_en:.1f} (acc={acc_pos_en:.1f}%, true_LM={lm_en.score_str(ctrl_en):.1f})")
print(f"  NegCtrl shuf   : {neg_en:.1f} ± {np.std(neg_scores):.1f}")

# ==============================================================================
# F1: ALTERNATING 180-DEG FLIP (1 CHAR SUR 2) — GLOBAL, PAR LIGNE, PHASES, POINT
# ==============================================================================
print("\n" + "=" * 86)
print("F1 : INVERSION 180° UN CARACTERE SUR DEUX (GLOBAL, 4 PHASES DE LIGNES, BASCULE AU POINT)")
print("=" * 86)

lines = [HAUER[:29], HAUER[29:60], HAUER[60:]]

def apply_alt180_global(seq, start_phase=0):
    return [rot180(s) if ((i + start_phase) % 2 == 1) else s for i, s in enumerate(seq)]

def apply_alt180_lines(lns, phases, dot_flip=False):
    out = []
    for li, ln in enumerate(lns):
        ph = phases[li]
        for idx, s in enumerate(ln):
            p = ph
            if dot_flip and li == 2 and idx >= 5:
                p ^= 1
            out.append(rot180(s) if ((idx + p) % 2 == 1) else s)
    return out

# First: check how alternating-180 affects a REAL English MASC if we encode it with alt-180!
# Positive control: take ctrl_en (87 chars), map letters to 24 (o,a), flip odd chars by 180, then undo & solve
rng_c = random.Random(55)
all24 = [f"{O[o]}{a}" for o in range(8) for a in (1, 2, 3)]
u_ctrl = list(set(ctrl_en))
mp_ctrl = dict(zip(u_ctrl, rng_c.sample(all24, len(u_ctrl))))
enc_ctrl = [mp_ctrl[c] for c in ctrl_en]
enc_ctrl_alt = apply_alt180_global(enc_ctrl, 0)
sc_raw_broken, _, _ = solve_masc(enc_ctrl_alt, lm_en, restarts=16, steps=6000, seed=56)
unflipped_ctrl = apply_alt180_global(enc_ctrl_alt, 0)
sc_restored, dec_restored, _ = solve_masc(unflipped_ctrl, lm_en, restarts=24, steps=8000, seed=57)
acc_restored = 100.0 * sum(a == b for a, b in zip(ctrl_en, dec_restored)) / 87.0
print(f"Contrôle positif (EN encodé + Alt-180°) : brut={sc_raw_broken:.1f} (uniq={len(set(enc_ctrl_alt))}, IC={ic_of(enc_ctrl_alt):.4f}) "
      f"-> après inversion Alt-180° : {sc_restored:.1f} (acc={acc_restored:.1f}%, IC={ic_of(unflipped_ctrl):.4f})")

f1_variants = [
    ("Global_pair",      apply_alt180_global(HAUER, 0)),
    ("Lignes_(0,0,0)",   apply_alt180_lines(lines, (0, 0, 0), False)),
    ("Lignes_(0,0,1)",   apply_alt180_lines(lines, (0, 0, 1), False)),
    ("Lignes_(0,1,0)",   apply_alt180_lines(lines, (0, 1, 0), False)),
    ("Lignes_(0,1,1)",   apply_alt180_lines(lines, (0, 1, 1), False)),
    ("Global+DotFlip",   apply_alt180_lines(lines, (0, 1, 0), True)),  # global phase is L1:0, L2:29%2=1, L3:60%2=0
    ("Lignes000+DotFlp", apply_alt180_lines(lines, (0, 0, 0), True)),
]

print(f"\n{'Variante F1':<18} {'Uniq':>4} {'IC':>7} {'Dbl':>4} {'180°':>5} {'EN_STD':>8} {'EN_BACK':>8} {'FR_STD':>8}  Extrait EN_STD")
print("-" * 102)
for vname, seq in f1_variants:
    dbl = sum(a == b for a, b in zip(seq, seq[1:]))
    m180 = sum((parse(b)[0] - parse(a)[0]) % 8 == 4 for a, b in zip(seq, seq[1:]))
    sc_e, dec_e, _ = solve_masc(seq, lm_en, restarts=24, steps=8000, seed=200)
    sc_b, _, _     = solve_masc(seq, lm_bk, restarts=16, steps=6000, seed=201)
    sc_f, _, _     = solve_masc(seq, lm_fr, restarts=16, steps=6000, seed=202)
    print(f"{vname:<18} {len(set(seq)):>4d} {ic_of(seq):>7.4f} {dbl:>4d} {m180:>5d} {sc_e:>8.1f} {sc_b:>8.1f} {sc_f:>8.1f}  {dec_e[:40]}")

# Also test Elgar's 1920s Notebook wheel (F1_TripletOri & F2_OctetArc, all shifts/rotations) on Alt-180 global & L000!
alpha_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
best_nb_alt = (-1e9, "", "")
for vname, seq in f1_variants[:3]:
    oas = [parse(s) for s in seq]
    for fam in ("F1_Triplet", "F2_Octet"):
        for cw in (1, -1):
            for o0 in range(8):
                for sh in range(24):
                    al = alpha_24[sh:] + alpha_24[:sh]
                    txt = "".join(al[((o - o0) * cw % 8) * 3 + (a - 1)] if fam == "F1_Triplet"
                                  else al[(a - 1) * 8 + ((o - o0) * cw % 8)] for o, a in oas)
                    sc = lm_en.score_str(txt)
                    if sc > best_nb_alt[0]:
                        best_nb_alt = (sc, f"{vname}|{fam}|cw={cw}|o0={o0}|sh={sh}", txt)
print(f"Meilleure clé Carnet 1920 sur Alt-180° (2,304 clés) : {best_nb_alt[0]:.1f} | {best_nb_alt[1]}")
print(f"  -> {best_nb_alt[2][:65]}")

# ==============================================================================
# F2: GEOMETRICALLY CONSTRAINED 180-DEG MIRROR HOMOPHONY (EXACT PAIRS ONLY!)
# ==============================================================================
print("\n" + "=" * 86)
print("F2 : HOMOPHONIE GEOMETRIQUE A 180° (FUSION UNIQUEMENT DE PAIRES MIROIR EXACTES X <-> rot180(X))")
print("=" * 86)

counts = Counter(HAUER)
# List all 12 possible (axis 0..3, arc 1..3) mirror pairs present in Dorabella
mirror_pairs_present = []
for ax in range(4):
    for a in (1, 2, 3):
        s1 = fmt(ax, a)
        s2 = fmt(ax + 4, a)
        c1, c2 = counts[s1], counts[s2]
        if c1 > 0 and c2 > 0:
            mirror_pairs_present.append((c1 + c2, s1, s2, c1, c2))
mirror_pairs_present.sort(reverse=True)
print(f"Paires miroir 180° dont les DEUX membres apparaissent dans Dorabella ({len(mirror_pairs_present)} paires) :")
for tot, s1, s2, c1, c2 in mirror_pairs_present:
    print(f"  {s1} ({c1:2d}x) <-> {s2} ({c2:2d}x)  [total={tot:2d}]")

def merge_mirror_subset(seq, pairs_to_merge):
    mp = {}
    for s1, s2 in pairs_to_merge:
        mp[s2] = s1
    return [mp.get(s, s) for s in seq]

# Positive control for F2: take ctrl_en (87 chars), split TOP 3 letters into 180-deg mirror homophones!
# Then test whether searching subsets of mirror pairs recovers ctrl_en cleanly!
c_ctrl = Counter(ctrl_en)
top3 = [ch for ch, _ in c_ctrl.most_common(3)]
# Assign 3 axes for top3 homophones, rest to other unused symbols
rng_h = random.Random(77)
used_slots = set()
ctrl_hom_map = {}
hom_pairs_true = []
for idx_t, ch in enumerate(top3):
    s1 = fmt(idx_t, 2); s2 = fmt(idx_t + 4, 2)
    used_slots.add(s1); used_slots.add(s2)
    ctrl_hom_map[ch] = (s1, s2)
    hom_pairs_true.append((s1, s2))
rem_slots = [s for s in all24 if s not in used_slots]
rng_h.shuffle(rem_slots)
for ch in set(ctrl_en) - set(top3):
    ctrl_hom_map[ch] = (rem_slots.pop(),)
enc_hom = [rng_h.choice(ctrl_hom_map[ch]) for ch in ctrl_en]
# Solve BEFORE merging vs AFTER merging the true 3 mirror pairs
sc_hom_raw, dec_hom_raw, _ = solve_masc(enc_hom, lm_en, restarts=20, steps=7000, seed=78)
acc_hom_raw = 100.0 * sum(a == b for a, b in zip(ctrl_en, dec_hom_raw)) / 87.0
enc_hom_merged = merge_mirror_subset(enc_hom, hom_pairs_true)
sc_hom_mrg, dec_hom_mrg, _ = solve_masc(enc_hom_merged, lm_en, restarts=24, steps=8000, seed=79)
acc_hom_mrg = 100.0 * sum(a == b for a, b in zip(ctrl_en, dec_hom_mrg)) / 87.0
print(f"\nContrôle positif F2 (EN avec E/T/A dédoublés en paires miroir 180°, IC={ic_of(enc_hom):.4f}, uniq={len(set(enc_hom))}):")
print(f"  Sans fusion : score={sc_hom_raw:.1f} (acc={acc_hom_raw:.1f}%)")
print(f"  Avec fusion des 3 paires miroir (IC={ic_of(enc_hom_merged):.4f}) : score={sc_hom_mrg:.1f} (acc={acc_hom_mrg:.1f}%)")
print(f"    -> {dec_hom_mrg[:65]}")

# Exhaustive search over ALL subsets of size k = 1, 2, 3, 4 of the 8 present mirror pairs in Dorabella!
# (C(8,1)=8, C(8,2)=28, C(8,3)=56, C(8,4)=70 -> total 162 subsets! Fast with 8 restarts x 4000 steps + top-5 deep polish!)
pair_list = [(s1, s2) for _, s1, s2, _, _ in mirror_pairs_present]
print("\nRecherche exhaustive sur les sous-ensembles de k = 1..5 paires miroir fusionnées (Dorabella Hauer_2025):")
for k in (1, 2, 3, 4, 5):
    combs = list(itertools.combinations(pair_list, k))
    # For k=1..4 test all combs; for k=5 C(8,5)=56 also fast!
    best_k = []
    for sub in combs:
        mseq = merge_mirror_subset(HAUER, sub)
        sc, dec, _ = solve_masc(mseq, lm_en, restarts=6, steps=3500, seed=300 + k)
        best_k.append((sc, ic_of(mseq), len(set(mseq)), sub, dec, mseq))
    best_k.sort(key=lambda x: -x[0])
    # Deep polish on #1
    _, ic1, u1, sub1, _, mseq1 = best_k[0]
    sc_deep, dec_deep, _ = solve_masc(mseq1, lm_en, restarts=28, steps=8000, seed=400 + k)
    # Shuffle baseline of the exact same merged sequence (to measure overfitting from fewer unique symbols!)
    sh = list(mseq1); random.Random(500 + k).shuffle(sh)
    sc_sh, _, _ = solve_masc(sh, lm_en, restarts=16, steps=6000, seed=510 + k)
    lbl = "+".join(f"{a}/{b}" for a, b in sub1)
    print(f"  k={k} ({len(combs):2d} subsets) | IC={ic1:.4f} u={u1:2d} | Best EN={sc_deep:6.1f} (shuf={sc_sh:6.1f}, Δ={sc_deep-sc_sh:+5.1f}) | [{lbl}]")
    print(f"       -> {dec_deep[:70]}")

# Also test full 180-deg folding (ALL 8 pairs merged -> 12 symbols)
folded_all = merge_mirror_subset(HAUER, pair_list)
sc_fa, dec_fa, _ = solve_masc(folded_all, lm_en, restarts=20, steps=7000, seed=600)
sh_fa = list(folded_all); random.Random(601).shuffle(sh_fa)
sc_fa_sh, _, _ = solve_masc(sh_fa, lm_en, restarts=14, steps=6000, seed=602)
print(f"  ALL (8 paires)   | IC={ic_of(folded_all):.4f} u={len(set(folded_all)):2d} | Best EN={sc_fa:6.1f} (shuf={sc_fa_sh:6.1f}, Δ={sc_fa-sc_fa_sh:+5.1f})")
print(f"       -> {dec_fa[:70]}")

# ==============================================================================
# F3: STATEFUL / DIFFERENTIAL ORIENTATION (Δori) & ARC-TRIGGERED 180° TOGGLE
# ==============================================================================
print("\n" + "=" * 86)
print("F3 : ENCODAGE DIFFERENTIEL (Δori = ori[i] - ori[i-1]) & BASCULE A ETAT 180°")
print("=" * 86)

oas = [parse(s) for s in HAUER]
f3_variants = []

# Differential orientation + current arc
f3_variants.append(("DeltaOri_Arc", [f"{(oas[i][0] - oas[i-1][0]) % 8}_{oas[i][1]}" for i in range(1, len(oas))]))
# Autokey Beaufort orientation (ori[i] + ori[i-1]) % 8 + current arc
f3_variants.append(("SumOri_Arc",   [f"{(oas[i][0] + oas[i-1][0]) % 8}_{oas[i][1]}" for i in range(1, len(oas))]))
# Differential orientation + differential arc
f3_variants.append(("DeltaOri_DArc",[f"{(oas[i][0] - oas[i-1][0]) % 8}_{(oas[i][1] - oas[i-1][1]) % 3}" for i in range(1, len(oas))]))

# Arc-triggered stateful 180-deg toggle (flip state AFTER seeing arc == trigger)
for trig in (1, 2, 3):
    state = 0
    seq_t = []
    for o, a in oas:
        seq_t.append(fmt(o + 4 * state, a))
        if a == trig:
            state ^= 1
    f3_variants.append((f"Toggle180_arc{trig}", seq_t))

# Flip 180 deg ONLY on specific arc class (e.g. only 1-arc, only 2-arc, or only 3-arc — note: preserves uniq count!)
# Wait: flipping all symbols of arc=k is a bijection on symbols -> MASC invariant! Only stateful/position-dependent matters.
# What about cumulative orientation (if Elgar added ori mod 8 as running key)? Already covered by DeltaOri_Arc!

print(f"{'Variante F3':<18} {'L':>3} {'Uniq':>4} {'IC':>7} {'EN_STD':>8} {'Shuf':>8} {'Δ':>6}  Extrait EN_STD")
print("-" * 92)
for vname, seq in f3_variants:
    sc_e, dec_e, _ = solve_masc(seq, lm_en, restarts=24, steps=8000, seed=700)
    sh = list(seq); random.Random(710).shuffle(sh)
    sc_sh, _, _ = solve_masc(sh, lm_en, restarts=14, steps=6000, seed=711)
    print(f"{vname:<18} {len(seq):>3d} {len(set(seq)):>4d} {ic_of(seq):>7.4f} {sc_e:>8.1f} {sc_sh:>8.1f} {sc_e-sc_sh:>+6.1f}  {dec_e[:42]}")

# ==============================================================================
# F4: PHYSICAL 180-DEG BOUSTROPHEDON (LINE 2 OR OTHER LINES ROTATED 180° + REVERSED)
# ==============================================================================
print("\n" + "=" * 86)
print("F4 : BOUSTROPHEDON PHYSIQUE A 180° (LIGNES RETOURNEES TETE-BECHE SUR LA CARTE)")
print("=" * 86)

def rot180_rev(ln):
    return [rot180(s) for s in ln[::-1]]

f4_variants = [
    ("L2_rot180+rev",     lines[0] + rot180_rev(lines[1]) + lines[2]),
    ("L2_rot180_only",    lines[0] + [rot180(s) for s in lines[1]] + lines[2]),
    ("L1+L3_rot180+rev",  rot180_rev(lines[0]) + lines[1] + rot180_rev(lines[2])),
    ("L3_rot180+rev",     lines[0] + lines[1] + rot180_rev(lines[2])),
    ("L3_afterDot_r180",  lines[0] + lines[1] + lines[2][:5] + [rot180(s) for s in lines[2][5:]]),
    ("L3_afterDot_r+rev", lines[0] + lines[1] + lines[2][:5] + rot180_rev(lines[2][5:])),
]
print(f"{'Variante F4':<18} {'Uniq':>4} {'IC':>7} {'EN_STD':>8} {'EN_BACK':>8}  Extrait EN_STD")
print("-" * 86)
for vname, seq in f4_variants:
    sc_e, dec_e, _ = solve_masc(seq, lm_en, restarts=24, steps=8000, seed=800)
    sc_b, _, _     = solve_masc(seq, lm_bk, restarts=16, steps=6000, seed=801)
    print(f"{vname:<18} {len(set(seq)):>4d} {ic_of(seq):>7.4f} {sc_e:>8.1f} {sc_b:>8.1f}  {dec_e[:46]}")

print(f"\nTemps total d'execution: {time.time() - t0:.1f} s")
