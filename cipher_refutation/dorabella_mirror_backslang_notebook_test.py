import itertools
import math
import random
import re
import time
from collections import Counter
import numpy as np

# Load clean helpers & expanded corpus without running piste_phonetic main block
with open("/home/user/dorabella_phonetic_transcriptions_test.py") as f:
    _base_code = f.read().split("# MAIN BENCHMARK EXECUTION")[0]
_ns = {}
exec(_base_code, _ns)

HAUER_2025 = _ns["HAUER_2025"]
HARTMEIER_2017 = _ns["HARTMEIER_2017"]
PELLING_2012 = _ns["PELLING_2012"]
SCHMEH_2018 = _ns["SCHMEH_2018"]
VISUAL_ALT = _ns["VISUAL_ALT"]
CORPUS_EN_FULL = _ns["CORPUS_EN_FULL"]
LangModel = _ns["LangModel"]
solve_masc = _ns["solve_masc"]
tokenize_words = _ns["tokenize_words"]
en_std = _ns["en_std"]
en_no_silent = _ns["en_no_silent"]
rev_lines = _ns["rev_lines"]
ic_of = _ns["ic_of"]

ORI_TO_IDX = {ch: i for i, ch in enumerate("ABCDEFGH")}
IDX_TO_ORI = "ABCDEFGH"

def parse_sym(s):
    return ORI_TO_IDX[s[0]], int(s[1])

def fmt_sym(o, a):
    return f"{IDX_TO_ORI[o]}{a}"

def mirror_sym(s):
    o, a = parse_sym(s)
    return fmt_sym((o + 4) % 8, a)

# Map Hartmeier letters back to (ori, arc) using Hartmeier's explicit rose:
# East(A)=A,B,C; SE(B)=D,E,F; S(C)=G,H,I; SW(D)=K,L,M; W(E)=N,O,P; NW(F)=Q,R,S; N(G)=T,U,W; NE(H)=X,Y,Z
HART_TO_OA = {}
_triplets = ["ABC", "DEF", "GHI", "KLM", "NOP", "QRS", "TUW", "XYZ"]
for oi, trip in enumerate(_triplets):
    for ai, ch in enumerate(trip, 1):
        HART_TO_OA[ch] = f"{IDX_TO_ORI[oi]}{ai}"
HARTMEIER_OA = [HART_TO_OA[ch] for ch in HARTMEIER_2017]

# Verify exact Pelling -> (ori, arc) map from Hauer_2025 majority + known 4 diffs
PELLING_OA = list(HAUER_2025)
PELLING_OA[12] = "G1"
PELLING_OA[21] = "G2"
PELLING_OA[77] = "B1"
PELLING_OA[84] = "C1"

# ==============================================================================
# BUILD LANGUAGE MODELS: EN_STD, EN_BACKSLANG, EN_BACKSLANG_PHON, EN_REV
# ==============================================================================
def en_backslang(text):
    return "".join(w[::-1] for w in tokenize_words(text))

def en_backslang_no_silent(text):
    keep_final_e = {"BE", "HE", "ME", "WE", "SHE", "THE", "YE", "SEE", "FEE", "FREE", "TREE", "THREE"}
    out = []
    for w in tokenize_words(text):
        w = re.sub(r"^KN", "N", w)
        w = re.sub(r"^WR", "R", w)
        w = w.replace("PH", "F").replace("QU", "KW").replace("CK", "K").replace("X", "KS")
        w = re.sub(r"IGH", "I", w)
        w = re.sub(r"OUGH", "O", w)
        w = re.sub(r"GH", "", w)
        w = re.sub(r"C(?=[EIY])", "S", w).replace("C", "K")
        if len(w) >= 3 and w.endswith("E") and w not in keep_final_e and not w.endswith("EE"):
            w = w[:-1]
        w = re.sub(r"(.)\1+", r"\1", w)
        out.append(w[::-1])
    return "".join(out)

EN_TRAIN_RAW = CORPUS_EN_FULL[:-1500]
EN_TEST_RAW = CORPUS_EN_FULL[-1500:]

lm_std = LangModel(en_std(EN_TRAIN_RAW), "EN_STD")
lm_back = LangModel(en_backslang(EN_TRAIN_RAW), "EN_BACKSLANG")
lm_back_ph = LangModel(en_backslang_no_silent(EN_TRAIN_RAW), "EN_BACK_PHON")
lm_rev = LangModel(en_std(EN_TRAIN_RAW)[::-1], "EN_REV")

t0 = time.time()

# ==============================================================================
# PART 1: STATISTICAL ANALYSIS OF MIRRORED PAIRS (180 deg, SAME ARC COUNT)
# ==============================================================================
print("=" * 84)
print("PARTIE 1 : ANALYSE STATISTIQUE DES PAIRES EN MIROIR (Δori = 180°, MEME NOMBRE D'ARCS)")
print("=" * 84)

def analyze_mirrors(seq, name, n_mc=50000, seed=42):
    oas = [parse_sym(s) for s in seq]
    n = len(oas)
    # Count exact 180-deg same-arc pairs, 180-deg any-arc pairs, and full Δori table
    mirror_same = []
    mirror_any = 0
    dori_same = Counter()
    dori_all = Counter()
    for i in range(n - 1):
        o1, a1 = oas[i]
        o2, a2 = oas[i + 1]
        d = (o2 - o1) % 8
        dori_all[d] += 1
        if a1 == a2:
            dori_same[d] += 1
            if d == 4:
                line, pos = (1, i + 1) if i < 29 else ((2, i - 28) if i < 60 else (3, i - 59))
                mirror_same.append((i, f"L{line}#{pos}", seq[i], seq[i + 1]))
        if d == 4:
            mirror_any += 1

    # Also count non-overlapping mirrored pairs (greedy left-to-right)
    non_ov = 0
    i = 0
    while i < n - 1:
        o1, a1 = oas[i]
        o2, a2 = oas[i + 1]
        if a1 == a2 and (o2 - o1) % 8 == 4:
            non_ov += 1
            i += 2
        else:
            i += 1

    # Monte Carlo 1: random shuffles of the 87 Dorabella tokens
    rng = np.random.default_rng(seed)
    ori_arr = np.array([o for o, _ in oas], dtype=np.int8)
    arc_arr = np.array([a for _, a in oas], dtype=np.int8)
    mc_same = np.empty(n_mc, dtype=np.int16)
    mc_any = np.empty(n_mc, dtype=np.int16)
    mc_d0_same = np.empty(n_mc, dtype=np.int16)
    idx_base = np.arange(n)
    for k in range(n_mc):
        p = rng.permutation(n)
        oo = ori_arr[p]
        aa = arc_arr[p]
        d4 = ((oo[1:] - oo[:-1]) % 8 == 4)
        sa = (aa[1:] == aa[:-1])
        mc_same[k] = int(np.sum(d4 & sa))
        mc_any[k] = int(np.sum(d4))
        mc_d0_same[k] = int(np.sum((oo[1:] == oo[:-1]) & sa))

    p_same = float(np.mean(mc_same >= len(mirror_same)))
    p_any = float(np.mean(mc_any >= mirror_any))
    z_same = (len(mirror_same) - mc_same.mean()) / max(mc_same.std(), 1e-9)

    print(f"\n--- {name} ---")
    print(f"  Paires miroir exactes (Δori=180°, même arcs) : {len(mirror_same)} (non-chevauchantes: {non_ov})")
    print(f"    Attendu par hasard (50k shuffles)          : {mc_same.mean():.2f} ± {mc_same.std():.2f} | Z = {z_same:+.2f} | p = {p_same:.6f}")
    print(f"  Paires opposées 180° (tous arcs confondus)   : {mirror_any} (attendu: {mc_any.mean():.2f} ± {mc_any.std():.2f} | p = {p_any:.5f})")
    print(f"  Distribution complète Δori (0°..315°), [même arc / tous arcs] :")
    row_s = "    " + "  ".join(f"{d*45:3d}°:{dori_same[d]:2d}/{dori_all[d]:2d}" for d in range(8))
    print(row_s)
    print(f"  Liste des {len(mirror_same)} paires miroir exactes :")
    for idx, loc, s1, s2 in mirror_same:
        print(f"    pos {idx:2d}-{idx+1:2d} ({loc:>5s}): {s1}->{s2}")
    return mirror_same

m_hauer = analyze_mirrors(HAUER_2025, "Hauer_2025 (consensus)", n_mc=50000, seed=42)
m_hart = analyze_mirrors(HARTMEIER_OA, "Hartmeier_2017", n_mc=20000, seed=43)
m_pell = analyze_mirrors(PELLING_OA, "Pelling_2012", n_mc=20000, seed=44)

# Monte Carlo 2: How often does a RANDOM assignment of 24 Dorabella symbols to English letters
# produce >= 13 mirrored pairs on an 87-char real English text?
rng2 = random.Random(777)
en_pool = en_std(CORPUS_EN_FULL)
# Precompute English slices of length 87
en_slices = [en_pool[i:i+87] for i in range(0, len(en_pool) - 87, 37)]
all_24_oa = [(o, a) for o in range(8) for a in (1, 2, 3)]
en_mirror_counts = []
for trial in range(10000):
    sl = en_slices[trial % len(en_slices)]
    # Map the <=24 unique letters in sl randomly to 24 (ori, arc) symbols
    uniq_l = list(set(sl))
    if len(uniq_l) > 24:
        # merge rarest letters like I=J, U=V
        sl = sl.replace("J", "I").replace("V", "U")
        uniq_l = list(set(sl))
    chosen = rng2.sample(all_24_oa, len(uniq_l))
    mp = dict(zip(uniq_l, chosen))
    cnt = 0
    prev = mp[sl[0]]
    for ch in sl[1:]:
        cur = mp[ch]
        if cur[1] == prev[1] and (cur[0] - prev[0]) % 8 == 4:
            cnt += 1
        prev = cur
    en_mirror_counts.append(cnt)
en_mc_arr = np.array(en_mirror_counts)
print(f"\nContrôle MASC anglais aléatoire (10k clés aléatoires sur tranches EN 87 car.):")
print(f"  Paires miroir attendues : {en_mc_arr.mean():.2f} ± {en_mc_arr.std():.2f} | max={en_mc_arr.max()} | P(>= {len(m_hauer)}) = {np.mean(en_mc_arr >= len(m_hauer)):.5f}")

# Check breakdown of mirrored pairs by pair type (which pairs occur?)
pair_types = Counter((s1, s2) for _, _, s1, s2 in m_hauer)
print("  Répartition par type de paire miroir (Hauer_2025):", dict(pair_types.most_common()))

# ==============================================================================
# PART 1B: TEST CRYPTOGRAPHIC HYPOTHESES ON MIRRORED PAIRS (Hauer et al. 2021)
# ==============================================================================
print("\n" + "=" * 84)
print("PARTIE 1B : TEST DES HYPOTHESES CRYPTOGRAPHIQUES SUR LES PAIRES MIROIR")
print("=" * 84)

# H_M1: Mirror pair (X, mirror(X)) encodes double letter XX -> restore (X, X)
def restore_mirror_doubles(seq):
    out = list(seq)
    i = 0
    while i < len(out) - 1:
        o1, a1 = parse_sym(out[i])
        o2, a2 = parse_sym(out[i + 1])
        if a1 == a2 and (o2 - o1) % 8 == 4:
            out[i + 1] = out[i]
            i += 2
        else:
            i += 1
    return out

# H_M1_alt: Mirror pair (X, mirror(X)) encodes YY (replace first by second)
def restore_mirror_doubles_rev(seq):
    out = list(seq)
    i = 0
    while i < len(out) - 1:
        o1, a1 = parse_sym(out[i])
        o2, a2 = parse_sym(out[i + 1])
        if a1 == a2 and (o2 - o1) % 8 == 4:
            out[i] = out[i + 1]
            i += 2
        else:
            i += 1
    return out

# H_M2: Second symbol of each mirrored pair is a NULL / dummy -> drop it
def drop_mirror_nulls(seq):
    out = []
    i = 0
    while i < len(seq):
        out.append(seq[i])
        if i < len(seq) - 1:
            o1, a1 = parse_sym(seq[i])
            o2, a2 = parse_sym(seq[i + 1])
            if a1 == a2 and (o2 - o1) % 8 == 4:
                i += 2
                continue
        i += 1
    return out

# H_M3: What if (X, mirror(X)) is ONLY applied to a single double-letter pair vs all?
# Also test folding 180-deg pairs globally (ori % 4 -> max 12 symbols)
def fold_180(seq):
    return [f"{IDX_TO_ORI[parse_sym(s)[0] % 4]}{parse_sym(s)[1]}" for s in seq]

# Positive control for H_M1: real 87-char English text where every double letter is replaced by mirror
ctrl_raw = en_std(EN_TEST_RAW)[120:207]
sc_ctrl_std, dec_ctrl_std, _ = solve_masc(list(ctrl_raw), lm_std, restarts=24, steps=8000, seed=11)
acc_ctrl_std = 100.0 * sum(a == b for a, b in zip(ctrl_raw, dec_ctrl_std)) / 87.0
print(f"Calibration EN_STD (87 car.): score={sc_ctrl_std:.1f} (acc={acc_ctrl_std:.1f}%)")

for label, fn in [
    ("M1a_RestoreXX (1er->2e)", restore_mirror_doubles),
    ("M1b_RestoreYY (2e->1er)", restore_mirror_doubles_rev),
    ("M2_DropMirrorNull",       drop_mirror_nulls),
]:
    tr_seq = fn(HAUER_2025)
    sc_e, dec_e, _ = solve_masc(tr_seq, lm_std, restarts=28, steps=8000, seed=21)
    sc_b, dec_b, _ = solve_masc(tr_seq, lm_back, restarts=28, steps=8000, seed=22)
    # Shuffle control of same transformed sequence
    sh = list(tr_seq)
    random.Random(99).shuffle(sh)
    sc_sh, _, _ = solve_masc(sh, lm_std, restarts=14, steps=6000, seed=23)
    # Normalize per char for fair comparison when len < 87
    print(f"  {label:<24} (L={len(tr_seq):2d}, u={len(set(tr_seq)):2d}, IC={ic_of(tr_seq):.4f}): "
          f"EN={sc_e:6.1f} ({sc_e/len(tr_seq):.3f}/c, shuf={sc_sh:6.1f}) | BACK={sc_b:6.1f}")
    print(f"    EN dec  : {dec_e}")

# ==============================================================================
# PART 2: WORD-BY-WORD BACKSLANG HYPOTHESIS (CALIBRATED)
# ==============================================================================
print("\n" + "=" * 84)
print("PARTIE 2 : HYPOTHESE DU BACKSLANG MOT PAR MOT (EX: CIPHER -> REHPIC)")
print("=" * 84)

print(f"Exemple EN_BACKSLANG : {en_backslang(EN_TEST_RAW)[:95]}")
print(f"IC corpus EN_BACKSLANG = {ic_of(en_backslang(EN_TRAIN_RAW)):.4f} | EN_BACK_PHON = {ic_of(en_backslang_no_silent(EN_TRAIN_RAW)):.4f}")

# Calibrate EN_BACKSLANG on 5 unseen 87-char slices
te_back = en_backslang(EN_TEST_RAW)
te_back_ph = en_backslang_no_silent(EN_TEST_RAW)
for name_m, lm_m, te_m in [("EN_BACKSLANG", lm_back, te_back), ("EN_BACK_PHON", lm_back_ph, te_back_ph)]:
    accs, scs, trues = [], [], []
    step = max(1, (len(te_m) - 90) // 5)
    for k in range(5):
        sl = te_m[k * step : k * step + 87]
        sc, dec, _ = solve_masc(list(sl), lm_m, restarts=28, steps=8000, seed=300 + k)
        acc = 100.0 * sum(a == b for a, b in zip(sl, dec)) / 87.0
        accs.append(acc); scs.append(sc); trues.append(lm_m.score_str(sl))
    # Negative controls (4 shuffles of Hauer)
    negs = []
    for k in range(4):
        sh = list(HAUER_2025)
        random.Random(400 + k).shuffle(sh)
        sc, _, _ = solve_masc(sh, lm_m, restarts=14, steps=6000, seed=410 + k)
        negs.append(sc)
    pos_m = float(np.mean(scs))
    neg_m = float(np.mean(negs))
    print(f"\n--- Modèle {name_m} ---")
    print(f"  Calibration (5 tranches 87 car.): True_LM={np.mean(trues):.1f} | PosCtrl={pos_m:.1f} (acc={np.mean(accs):.1f}%, max={np.max(accs):.1f}%) | NegCtrl={neg_m:.1f}")
    for tname, seq in [("Hauer_2025", HAUER_2025), ("Pelling_2012", PELLING_2012), ("Visual_Alt5", VISUAL_ALT), ("Schmeh_2018", SCHMEH_2018)]:
        sc0, dec0, _ = solve_masc(seq, lm_m, restarts=28, steps=8000, seed=500)
        sc2, dec2, _ = solve_masc(rev_lines(seq), lm_m, restarts=28, steps=8000, seed=501)
        bsc, bdec, tag = (sc0, dec0, "T0") if sc0 >= sc2 else (sc2, dec2, "T2")
        print(f"  {tname:<14}: T0={sc0:6.1f} | T2={sc2:6.1f} | vs_Pos={bsc - pos_m:+6.1f} | vs_Neg={bsc - neg_m:+6.1f} [{tag}] {bdec[:52]}")

# ==============================================================================
# PART 3: ELGAR'S 1920s NOTEBOOK KEY ("MARCO ELGAR" / "A VERY OLD CYPHER")
#         ALL ROTATIONS, REFLECTIONS, ARC PERMS, CAESAR SHIFTS & ROTATING PIGPEN
# ==============================================================================
print("\n" + "=" * 84)
print("PARTIE 3 : CLE DU CARNET D'ELGAR (1920s) — ROTATIONS, MIROIRS, DECALAGES & PIGPEN ROTATIF")
print("=" * 84)

# First: score the exact Pelling/Gaffney Notebook string & Hartmeier Marco-Elgar string directly!
pelling_str = "".join(PELLING_2012)
hart_str = "".join(HARTMEIER_2017)
# Compare against random 87-char uniform alphabet score
rng3 = random.Random(1234)
rand_en = [lm_std.score_str("".join(rng3.choices("ABCDEFGHIKLMNOPQRSTUVWXYZ", k=87))) for _ in range(500)]
rand_bk = [lm_back.score_str("".join(rng3.choices("ABCDEFGHIKLMNOPQRSTUVWXYZ", k=87))) for _ in range(500)]
print(f"Référence bruit uniforme (24 lettres) : EN_STD={np.mean(rand_en):.1f}±{np.std(rand_en):.1f} | EN_BACK={np.mean(rand_bk):.1f}±{np.std(rand_bk):.1f}")
print(f"Référence vrai texte 87 car.          : EN_STD={lm_std.score_str(ctrl_raw):.1f} | EN_BACK={lm_back.score_str(te_back[:87]):.1f}")
print(f"Clé directe Pelling/Gaffney (BLTACE..) : EN_STD={lm_std.score_str(pelling_str):.1f} | EN_BACK={lm_back.score_str(pelling_str):.1f} | EN_REV={lm_rev.score_str(pelling_str):.1f}")
print(f"Clé directe Marco-Elgar/Hart (BPECAH..): EN_STD={lm_std.score_str(hart_str):.1f} | EN_BACK={lm_back.score_str(hart_str):.1f} | EN_REV={lm_rev.score_str(hart_str):.1f}")

# Exhaustive geometric sweep over BOTH Notebook families:
# Alphabet 24-letter reductions:
ALPHAS_24 = [
    ("IJ_UV", "ABCDEFGHIKLMNOPQRSTUWXYZ"), # standard 1920s notebook (I=J, U=V)
    ("IJ_VW", "ABCDEFGHIKLMNOPQRSTUVXYZ"), # alt (I=J, W=V)
    ("IJ_QX", "ABCDEFGHIKLMNOPRSTUVWYZ" + "Q"), # rare alt
]
# Wait: third should be 24 distinct letters (drop J and Q -> keep X, Z):
ALPHAS_24 = [
    ("IJ_UV", "ABCDEFGHIKLMNOPQRSTUWXYZ"),
    ("IJ_VW", "ABCDEFGHIKLMNOPQRSTUVXYZ"),
    ("IJ_QZ", "ABCDEFGHIKLMNOPRSTUVWXY" + "Z"), # drop J, Q
]
# Clean 24-letter alphabets:
ALPHAS_24 = [
    ("IJ_UV", "ABCDEFGHIKLMNOPQRSTUWXYZ"), # 24: no J, V
    ("IJ_VW", "ABCDEFGHIKLMNOPQRSTUVXYZ"), # 24: no J, W
    ("JZ_drop", "ABCDEFGHIKLMNOPQRSTUVWXY"), # 24: no J, Z
]

# Pre-parse Hauer_2025 and Pelling_OA
seqs_oa = {
    "Hauer": [parse_sym(s) for s in HAUER_2025],
    "Pelling": [parse_sym(s) for s in PELLING_OA],
}

best_hits = {m: [] for m in ["EN_STD", "EN_BACKSLANG", "EN_REV"]}
lm_dict = {"EN_STD": lm_std, "EN_BACKSLANG": lm_back, "EN_REV": lm_rev}

total_tested = 0
for sname, oas in seqs_oa.items():
    for aname, alpha0 in ALPHAS_24:
        for atbash in (False, True):
            alpha_dir = alpha0[::-1] if atbash else alpha0
            for shift in range(24):
                alpha = alpha_dir[shift:] + alpha_dir[:shift]
                for cw in (1, -1):
                    for start_ori in range(8):
                        for arc_perm in itertools.permutations([0, 1, 2]):
                            # Family 1: Triplet-by-orientation (Marco Elgar 1924: ori*3 + arc)
                            # Family 2: Octet-by-arc (Pelling/Gaffney: arc*8 + ori)
                            for fam in ("F1_TripletOri", "F2_OctetArc"):
                                chars = []
                                for o, a in oas:
                                    oo = ((o - start_ori) * cw) % 8
                                    aa = arc_perm[a - 1]
                                    idx = (oo * 3 + aa) if fam == "F1_TripletOri" else (aa * 8 + oo)
                                    chars.append(alpha[idx])
                                txt = "".join(chars)
                                total_tested += 1
                                for mname, lm in lm_dict.items():
                                    sc = lm.score_str(txt)
                                    if len(best_hits[mname]) < 3 or sc > best_hits[mname][-1][0]:
                                        best_hits[mname].append((sc, f"{sname}|{fam}|{aname}|atb={int(atbash)}|sh={shift}|cw={cw}|o0={start_ori}|ap={arc_perm}", txt))
                                        best_hits[mname].sort(key=lambda x: -x[0])
                                        best_hits[mname] = best_hits[mname][:3]

print(f"\nBalayage géométrique exhaustif (Famille 1 Marco-Elgar + Famille 2 Octet-Arc) : {total_tested:,} clés testées !")
for mname in ["EN_STD", "EN_BACKSLANG", "EN_REV"]:
    print(f"  Top 3 sous {mname} (vrai texte ~ -205, bruit ~ -365) :")
    for sc, desc, txt in best_hits[mname]:
        print(f"    {sc:6.1f} | {desc}")
        print(f"           -> {txt}")

# Family 3: Rotating Pigpen (Alberti wheel / Courage-card progressive rotation!)
# ori'(i) = (ori(i) + step*i + off) % 8, arc'(i) = (arc(i)-1 + astep*i) % 3
rot_tested = 0
best_rot = {m: [] for m in ["EN_STD", "EN_BACKSLANG"]}
alpha_std = "ABCDEFGHIKLMNOPQRSTUWXYZ"
for oas_name, oas in [("Hauer", seqs_oa["Hauer"])]:
    for fam in ("F1_TripletOri", "F2_OctetArc"):
        for step in range(1, 8):
            for astep in (0, 1, 2):
                for cw in (1, -1):
                    for start_ori in range(8):
                        for shift in range(24):
                            alpha = alpha_std[shift:] + alpha_std[:shift]
                            chars = []
                            for pos, (o, a) in enumerate(oas):
                                oo = ((o - start_ori + step * pos) * cw) % 8
                                aa = (a - 1 + astep * pos) % 3
                                idx = (oo * 3 + aa) if fam == "F1_TripletOri" else (aa * 8 + oo)
                                chars.append(alpha[idx])
                            txt = "".join(chars)
                            rot_tested += 1
                            for mname in ("EN_STD", "EN_BACKSLANG"):
                                sc = lm_dict[mname].score_str(txt)
                                if len(best_rot[mname]) < 3 or sc > best_rot[mname][-1][0]:
                                    best_rot[mname].append((sc, f"{fam}|step={step}|astep={astep}|cw={cw}|o0={start_ori}|sh={shift}", txt))
                                    best_rot[mname].sort(key=lambda x: -x[0])
                                    best_rot[mname] = best_rot[mname][:3]

# Also test the exact Courage-card length-10 progressive offset (+0,+1,+2,+3,+4,+5,+6,+7,0,0) on ori and (0..0,-1,-2) on arc
courage_ori = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0]
courage_arc = [0, 0, 0, 0, 0, 0, 0, 0, -1, -2]
best_courage = (-1e9, "", "")
for fam in ("F1_TripletOri", "F2_OctetArc"):
    for sign in (1, -1):
        for cw in (1, -1):
            for start_ori in range(8):
                for shift in range(24):
                    alpha = alpha_std[shift:] + alpha_std[:shift]
                    chars = []
                    for pos, (o, a) in enumerate(seqs_oa["Hauer"]):
                        oo = ((o - start_ori + sign * courage_ori[pos % 10]) * cw) % 8
                        aa = (a - 1 + sign * courage_arc[pos % 10]) % 3
                        idx = (oo * 3 + aa) if fam == "F1_TripletOri" else (aa * 8 + oo)
                        chars.append(alpha[idx])
                    txt = "".join(chars)
                    rot_tested += 1
                    sc = lm_std.score_str(txt)
                    if sc > best_courage[0]:
                        best_courage = (sc, f"Courage10|{fam}|sign={sign}|cw={cw}|o0={start_ori}|sh={shift}", txt)

print(f"\nBalayage Pigpen Rotatif (Alberti + Carte Courage 10-pas) : {rot_tested:,} clés testées !")
for mname in ("EN_STD", "EN_BACKSLANG"):
    print(f"  Top 2 Rotatif sous {mname} :")
    for sc, desc, txt in best_rot[mname][:2]:
        print(f"    {sc:6.1f} | {desc} -> {txt[:62]}")
print(f"  Meilleur Carte Courage (période 10) sous EN_STD : {best_courage[0]:6.1f} | {best_courage[1]}")
print(f"    -> {best_courage[2][:68]}")

print(f"\nTemps total d'execution: {time.time() - t0:.1f} s")
