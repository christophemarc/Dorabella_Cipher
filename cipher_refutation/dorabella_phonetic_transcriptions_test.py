import math
import random
import re
import time
import unicodedata
from collections import Counter
import numpy as np

# ==============================================================================
# 1. FOUR PUBLISHED TRANSCRIPTIONS + VISUAL-ALT VARIANT
# ==============================================================================
HAUER_2025 = (
    "A2 E3 B2 A3 A1 C2 G1 A3 D1 H2 B3 F2 F1 B1 F2 C3 F2 F2 C2 E3 E3 F2 B1 H1 H2 H1 C1 B3 F3 "
    "G1 F2 G1 C2 H1 A3 D1 D2 A3 B2 F2 F2 B2 C2 C1 F1 G1 F2 B3 F2 C2 G2 F3 F1 B1 H1 D1 D1 H1 B3 F3 "
    "B2 F3 C2 G2 F3 B2 B1 G2 G3 C1 F3 B2 F2 C2 G2 F1 F3 C1 A3 E3 C1 F3 C2 A3 B1 H1 A3"
).split()

SCHMEH_2018 = list(
    "ABCDEFGDHAIJKLJMJJFBBJNGOGNIP"
    "GJGFQDHRSCJJCFNKGJIJFTPKLQHHQIP"
    "CPFUPCLUUNPCJFUKPNDBNPFDLED"
)

HARTMEIER_2017 = list(
    "BPECAHTCKYFRQDRIRRHPPRDXYXGFS"
    "TRTHTCKLCERREHGQTRFRHUSQDXKKXFS"
    "ESHUSEDUWGSERHUQSDCPGSHCDXC"
)

PELLING_2012 = list(
    "BLTACEIARWUNISNFNNELLHSYWYDUO"
    "INIEYARQATNNTEDMINUNEHOMSYRRYUO"
    "TOEHOTSHGDOTNEHMOSALDOEADYA"
)

# Visual-Alt: applies the 5 most visually plausible single-step (+/-45 deg) flips
# observed on the facsimile & disputed between Hartmeier/Pelling/Schmeh:
#   idx 12 (L1#13): F1 -> G1 (Pelling)
#   idx 21 (L1#22): F2 -> G2 (Pelling)
#   idx 33 (L2#5):  H1 -> G1 (Hartmeier)
#   idx 77 (L3#18): C1 -> B1 (BOTH Hartmeier & Pelling)
#   idx 85 (L3#26): H1 -> A1 (Schmeh — visually very close to L1#5 A1)
VISUAL_ALT = list(HAUER_2025)
VISUAL_ALT[12] = "G1"
VISUAL_ALT[21] = "G2"
VISUAL_ALT[33] = "G1"
VISUAL_ALT[77] = "B1"
VISUAL_ALT[85] = "A1"

TRANSCRIPTIONS = {
    "Hauer_2025": HAUER_2025,
    "Hartmeier_2017": HARTMEIER_2017,
    "Pelling_2012": PELLING_2012,
    "Schmeh_2018": SCHMEH_2018,
    "Visual_Alt5": VISUAL_ALT,
}

# ==============================================================================
# 2. STRUCTURAL METRICS (IC, SUKHOTIN VOWELS, DOUBLES, REVERSALS, LAG-31)
# ==============================================================================
def ic_of(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    c = Counter(seq)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))

def sukhotin_vowels(seq):
    syms = sorted(set(seq))
    idx = {s: i for i, s in enumerate(syms)}
    k = len(syms)
    M = np.zeros((k, k), dtype=float)
    for a, b in zip(seq, seq[1:]):
        if a != b:
            M[idx[a], idx[b]] += 1
            M[idx[b], idx[a]] += 1
    row_sums = M.sum(axis=1)
    vowels = set()
    while True:
        best_i = -1
        best_val = 0.0
        for i in range(k):
            if i not in vowels and row_sums[i] > best_val:
                best_val = row_sums[i]
                best_i = i
        if best_i == -1:
            break
        vowels.add(best_i)
        for j in range(k):
            row_sums[j] -= 2.0 * M[best_i, j]
    v_syms = {syms[i] for i in vowels}
    v_rate = sum(1 for x in seq if x in v_syms) / len(seq)
    vc_alt = sum(1 for a, b in zip(seq, seq[1:]) if (a in v_syms) != (b in v_syms)) / (len(seq) - 1)
    return v_syms, v_rate, vc_alt

def seq_stats(seq):
    n = len(seq)
    uniq = len(set(seq))
    ic = ic_of(seq)
    v_syms, v_rate, vc_alt = sukhotin_vowels(seq)
    doubles = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    digrams = set(zip(seq, seq[1:]))
    rev_digrams = sum(1 for (a, b) in digrams if a < b and (b, a) in digrams)
    lag31 = sum(1 for i in range(n - 31) if seq[i] == seq[i + 31])
    return {
        "uniq": uniq,
        "ic": ic,
        "suk_v": v_rate,
        "vc_alt": vc_alt,
        "doubles": doubles,
        "rev_digrams": rev_digrams,
        "lag31": lag31,
    }

# ==============================================================================
# 3. EXPANDED ENGLISH & FRENCH CORPORA (WITH SPACES FOR WORD-LEVEL RULES)
# ==============================================================================
import sys
sys.path.insert(0, "/home/user")
from dorabella_piste1_test import CORPUS_EN as BASE_EN

EXTRA_EN = """
Edward Elgar composed his Variations on an Original Theme popularly known as the Enigma Variations between October eighteen ninety eight and February eighteen ninety nine. It is an orchestral work comprising fourteen variations on an original theme. After its premiere in London the variations quickly achieved popularity and helped establish Elgars growing reputation in Britain and internationally. It is now a staple of the orchestral repertoire globally.
Elgar dedicated the work to my friends pictured within each variation being a musical sketch of or a musical idea related to one of his circle. Those musically sketched include Elgars wife Alice his friend and publisher August Jaeger and in the final variation Elgar himself.
In addition to the miniature depictions of his friends Elgar said that the main theme which he called Enigma referred in some way to a larger unspecified theme. By doing so he posed a challenge that has generated much speculation but has never been conclusively answered. The Enigma theme is widely believed to involve a hidden melody although some commentators have taken it to represent an abstract idea rather than a musical theme.
Elgar was a largely self taught composer with no influential patrons or sponsors and from the seventies to the late nineties he struggled for artistic recognition and financial success. He supplemented the modest income from his early compositions by giving violin lessons. During the nineties he composed the concert overture Froissart and a Serenade for Strings but such modest fame as he acquired by the end of the decade derived mainly from four large scale works for chorus and orchestra The Black Knight The Light of Life King Olaf and Caractacus.
Elgar described how on the evening of twenty first October after a tiring day teaching he sat down at the piano. A melody he played caught the attention of his wife and he began to improvise variations on it in styles which reflected the character of some of his friends. These improvisations expanded and orchestrated became the Enigma Variations.
In a letter to his close friend his publisher at Novello August Jaeger Elgar wrote I have sketched a set of Variations on an original theme the Variations have amused me because I have labelled them with the nicknames of my particular friends you are Nimrod. That is to say I have written the variations each one to represent the mood of the party I have liked to imagine the party writing the variation himself or herself and have written what I think they would have written if they were asses enough to compose. It is a quaint idea and the result is amusing to those behind the scenes and will not affect the hearer who knows nothing. What think you.
The variations were first performed at Saint James Hall in London conducted by Hans Richter. Elgar was twice called onto the platform to acknowledge the enthusiastic applause. A few critics complained about the layer of mystification but most praised the substance structure and orchestration of the work.
Jaeger felt that the final variation depicting Elgar himself was too short to make its full effect and persuaded the extremely reluctant composer who prided himself on never having to alter any of his compositions to enlarge it. Ninety six bars and an organ part were added for the third performance of the variations. The new version which is usually played today was first heard at the Worcester Three Choirs Festival with Elgar conducting.
This work commenced in a spirit of humour and continued in deep seriousness contains sketches of the composers friends. It may be understood that these personages comment or reflect on the original theme and each one attempts a solution of the Enigma for so the theme is called. The sketches are not portraits but each variation contains a distinct idea founded on some particular personality or perhaps on some incident known only to two people. This is the basis of the composition but the work may be listened to as a piece of music apart from any extraneous consideration.
Pride and Prejudice is a novel by English author Jane Austen. Written when she was aged twenty it was her third novel written and became the second to see print when it was published in eighteen thirteen. A novel of manners it follows the character development of Elizabeth Bennet the protagonist of the book who learns about the repercussions of hasty judgments and comes to appreciate the difference between superficial goodness and actual goodness.
Her father Mister Bennet owner of the Longbourn estate in Hertfordshire has five daughters but this estate is entailed and so can only be inherited in the male line. To his regret he has failed to save out of the income from the Longbourn estate to provide for his daughters. From the Bennets perspective it is imperative that at least one of their daughters marry well to support the others which is a primary motivation driving the plot.
In the early nineteenth century the Bennet family lives at their Longbourn estate situated near the small town of Meryton in Hertfordshire England. Missus Bennets greatest desire is to marry off her five daughters to secure their futures. A regiment of militia is encamped outside the town and the younger Bennet daughters flirt with the officers to the great embarrassment of their elder sisters.
The arrival of Mister Bingley a rich bachelor who rents the neighbouring Netherfield estate gives Missus Bennet hope that one of her daughters might contract a marriage to their advantage. Soon the family is introduced to the Netherfield party Mister Bingley his sisters and brother in law and his closest friend Mister Darcy. Mister Bingley appears interested in Jane. Mister Darcy declines to dance with Elizabeth as she is not handsome enough. Although she jokes about it Elizabeth is deeply offended. Despite this Mister Darcy secretly finds himself drawn to Elizabeth as they continue to encounter each other at social events appreciating her wit and frankness.
Mister Collins a recently ordained clergyman and as a distant cousin the heir to the Longbourn estate which is entailed away from the daughters visits the Bennet family. He intends to marry one of the Bennet girls on the advice of his patroness Lady Catherine de Bourgh who is also Mister Darcys aunt. He decides to pursue Elizabeth. The Bennet family meets the charming militia officer George Wickham who tells Elizabeth that Mister Darcy had treated him badly in the past. Elizabeths prejudice against Mister Darcy and admiration for Wickham lead her to believe him. Elizabeth rejects Mister Collinss marriage proposal and Mister Collins subsequently proposes to Charlotte Lucas a friend of Elizabeths and is accepted. Convinced that Jane is an unsuitable match for his friend Mister Darcy prompts Mister Bingley to leave for London and to never return to Netherfield. Jane is heartbroken while Elizabeths hatred for Mister Darcy grows.
"""

CORPUS_EN_FULL = BASE_EN + "\n" + EXTRA_EN

# Load CORPUS_FR directly by parsing the file without executing piste3 import
with open("/home/user/dorabella_fr_ga_test.py", "r") as f:
    src_fr = f.read()
m_fr = re.search(r'CORPUS_FR\s*=\s*"""(.*?)"""', src_fr, re.S)
CORPUS_FR_FULL = m_fr.group(1)

# ==============================================================================
# 4. FIVE DETERMINISTIC PHONETIC / SHORTHAND TRANSFORMS
# ==============================================================================
def tokenize_words(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").upper()
    return re.findall(r"[A-Z]+", text)

def en_std(text):
    return "".join(tokenize_words(text))

# Regime A: Victorian phonetic/reformed spelling (silent-e & silent consonants & doubles dropped)
def en_no_silent(text):
    keep_final_e = {"BE", "HE", "ME", "WE", "SHE", "THE", "YE", "SEE", "FEE", "FREE", "TREE", "THREE"}
    out = []
    for w in tokenize_words(text):
        # Silent consonant clusters & phonetic respellings
        w = re.sub(r"^KN", "N", w)
        w = re.sub(r"^WR", "R", w)
        w = re.sub(r"^GN", "N", w)
        w = re.sub(r"^PS", "S", w)
        w = w.replace("PH", "F")
        w = w.replace("QU", "KW")
        w = w.replace("CK", "K")
        w = w.replace("X", "KS")
        w = re.sub(r"IGH", "I", w)
        w = re.sub(r"OUGH", "O", w)
        w = re.sub(r"AUGH", "A", w)
        w = re.sub(r"GH", "", w)
        w = re.sub(r"MB$", "M", w)
        w = re.sub(r"BT", "T", w)
        w = re.sub(r"C(?=[EIY])", "S", w)
        w = w.replace("C", "K")
        # Silent final E (for words >= 4 chars, or >= 3 not in keep_final_e)
        if len(w) >= 3 and w.endswith("E") and w not in keep_final_e and not w.endswith("EE"):
            w = w[:-1]
        # Silent E in -ED / -ES after consonant (e.g. PLAYED->PLAYD, LOVES->LOVS)
        w = re.sub(r"([BCDFGHJKLMNPQRSTVWXZ])ED$", r"\1D", w)
        w = re.sub(r"([BCDFGHJKLMNPQRSTVWXZ])ES$", r"\1S", w)
        # Collapse double letters
        w = re.sub(r"(.)\1+", r"\1", w)
        out.append(w)
    return "".join(out)

# Regime B: True 22-24 phoneme shorthand (1 symbol = 1 phoneme, digraphs merged into single symbols)
# Uses ASCII symbols:
#   Consonants: P B T D K G F V S Z M N L R W Y H + Q(=TH) C(=SH/ZH) J(=CH/J) X(=NG)  (21 consonants -> merged Y/W or H if needed)
#   Vowels: A E I O U (5 core vowel classes) -> total alphabet <= 24 (actually we merge J->C or Y->I so <= 24 symbols!)
def en_phonemic_24(text):
    lex = {
        "THE": "QE", "AND": "AND", "OF": "OV", "TO": "TU", "IN": "IN", "IS": "IZ",
        "WAS": "WOZ", "WERE": "WER", "ARE": "AR", "YOU": "IU", "YOUR": "IOR",
        "HAVE": "HAV", "HAS": "HAZ", "HAD": "HAD", "ONE": "WON", "TWO": "TU",
        "WHO": "HU", "WHAT": "WOT", "WHICH": "WIC", "THAT": "QAT", "THIS": "QIS",
        "WITH": "WIQ", "THEY": "QE", "THEIR": "QER", "THEM": "QEM", "FROM": "FROM",
        "WOULD": "WUD", "COULD": "KUD", "SHOULD": "CUD", "THROUGH": "QRU",
        "THOUGH": "QO", "THOUGHT": "QOT", "ENOUGH": "INUF", "LAUGH": "LAF",
        "KNOW": "NO", "KNOWN": "NON", "WRITE": "RIT", "WRITTEN": "RITN",
        "MUSIC": "MIUZIK", "COMPOSER": "KOMPOZR", "VARIATIONS": "VERIECNZ",
        "ENGLISH": "IXGLIC", "FRIEND": "FREND", "FRIENDS": "FRENDZ",
        "WORK": "WERK", "WORKS": "WERKS", "SOME": "SUM", "COME": "KUM",
        "DONE": "DUN", "LOVE": "LUV", "MOTHER": "MUQR", "FATHER": "FAQR",
        "BROTHER": "BRUQR", "OTHER": "UQR", "ANOTHER": "ANUQR",
    }
    out2 = []
    for w in tokenize_words(text):
        if w in lex:
            out2.append(lex[w])
            continue
        w = re.sub(r"TION$", "1N", w)
        w = re.sub(r"SION$", "1N", w)
        w = re.sub(r"CIAL$", "1L", w)
        w = re.sub(r"TIAL$", "1L", w)
        w = re.sub(r"^KN", "N", w)
        w = re.sub(r"^WR", "R", w)
        w = re.sub(r"^GN", "N", w)
        w = re.sub(r"^WH", "W", w)
        w = w.replace("TCH", "1").replace("CH", "1").replace("SH", "1")
        w = w.replace("TH", "2")
        w = w.replace("NG", "3")
        w = w.replace("PH", "F").replace("QU", "KW").replace("CK", "K")
        w = w.replace("X", "KS")
        w = re.sub(r"IGH", "I", w)
        w = re.sub(r"OUGH", "O", w)
        w = re.sub(r"AUGH", "O", w)
        w = re.sub(r"GH", "", w)
        w = re.sub(r"MB$", "M", w)
        w = re.sub(r"C(?=[EIY])", "S", w)
        w = w.replace("C", "K")
        w = w.replace("J", "1")  # merge affricate/palatal into 1 (keeps <= 23 symbols)
        # Vowel digraphs -> single core vowel
        w = re.sub(r"EE|EA|IE|EI", "I", w)
        w = re.sub(r"OO|EW|UE|UI", "U", w)
        w = re.sub(r"AI|AY", "E", w)
        w = re.sub(r"OA|OW|AW|AU|OU", "O", w)
        w = re.sub(r"OI|OY", "O", w)
        # Final silent E and weak endings
        if len(w) >= 3 and w.endswith("E"):
            w = w[:-1]
        w = re.sub(r"([BDFGKLMNPRSTVWZ123])E([DNRS])$", r"\1\2", w)
        w = w.replace("Y", "I")  # Y -> I (keeps alphabet at 23 symbols: A E I O U + 18 cons)
        w = w.replace("1", "C").replace("2", "Q").replace("3", "X")
        w = re.sub(r"(.)\1+", r"\1", w)
        out2.append(w)
    return "".join(out2)

# Regime C: Vowel-reduced shorthand (silent & unstressed internal vowels dropped, initial vowels kept)
def en_shorthand_drop_v(text):
    out = []
    vowels = set("AEIOU")
    for w in tokenize_words(text):
        w = re.sub(r"^KN", "N", w)
        w = re.sub(r"^WR", "R", w)
        w = w.replace("PH", "F").replace("CK", "K").replace("QU", "KW")
        w = re.sub(r"GH", "", w)
        w = re.sub(r"C(?=[EIY])", "S", w)
        w = w.replace("C", "K")
        w = re.sub(r"(.)\1+", r"\1", w)
        if len(w) <= 2:
            out.append(w)
            continue
        # Keep first char; drop final E; keep at most first vowel inside word, drop subsequent internal vowels
        res = [w[0]]
        kept_v = w[0] in vowels
        core = w[1:-1] if w.endswith("E") else w[1:]
        for ch in core:
            if ch in vowels:
                if not kept_v:
                    res.append(ch)
                    kept_v = True
            else:
                res.append(ch)
        w2 = re.sub(r"(.)\1+", r"\1", "".join(res))
        out.append(w2)
    return "".join(out)

# Regime D: Consonant-skeleton / unvocalized Pitman-style abjad (only consonants + word-initial vowels)
def en_abjad(text):
    out = []
    vowels = set("AEIOUY")
    for w in tokenize_words(text):
        w = re.sub(r"^KN", "N", w).replace("PH", "F").replace("CK", "K")
        w = re.sub(r"GH", "", w)
        w = re.sub(r"C(?=[EIY])", "S", w).replace("C", "K")
        res = [w[0]] + [ch for ch in w[1:] if ch not in vowels]
        w2 = re.sub(r"(.)\1+", r"\1", "".join(res))
        out.append(w2)
    return "".join(out)

# Regime E: French phonetic shorthand (silent final -e, -s, -t, -d, -x, -ent dropped; digraphs collapsed)
def fr_phonetic(text):
    out = []
    for w in tokenize_words(text):
        w = w.replace("H", "")
        w = w.replace("EAU", "O").replace("AU", "O")
        w = w.replace("OU", "U")
        w = w.replace("CH", "1")
        w = w.replace("GN", "2")
        w = w.replace("PH", "F")
        w = w.replace("QU", "K")
        w = w.replace("AI", "E").replace("EI", "E")
        w = w.replace("OI", "WA")
        w = w.replace("EU", "E").replace("OE", "E")
        w = re.sub(r"C(?=[EIY])", "S", w)
        w = w.replace("C", "K")
        w = w.replace("X", "KS")
        w = w.replace("Y", "I")
        # Silent endings
        if len(w) >= 4 and w.endswith("ENT"):
            w = w[:-2]  # keep N or drop ENT -> conservative: drop NT -> keep E? No, 3pl -ent silent! Drop ENT if >4
            if len(w) >= 4:
                w = w[:-1]
        if len(w) >= 3 and w.endswith("ES"):
            w = w[:-2]
        elif len(w) >= 3 and w[-1] in "ESTDXZ":
            w = w[:-1]
        w = w.replace("1", "C").replace("2", "Q")
        w = re.sub(r"(.)\1+", r"\1", w)
        if w:
            out.append(w)
    return "".join(out)

# ==============================================================================
# 5. FAST INTERPOLATED TRIGRAM LANGUAGE MODEL + SA/2-OPT MASC SOLVER
# ==============================================================================
class LangModel:
    def __init__(self, text, name):
        self.name = name
        self.text = text
        self.alpha = sorted(set(text))
        self.V = len(self.alpha)
        self.c2i = {c: i for i, c in enumerate(self.alpha)}
        self.i2c = {i: c for i, c in enumerate(self.alpha)}

        ids = [self.c2i[c] for c in text]
        n = len(ids)
        u = np.ones(self.V) * 0.5
        b = np.zeros((self.V, self.V))
        t = np.zeros((self.V, self.V, self.V))
        for i in range(n):
            u[ids[i]] += 1
        for i in range(n - 1):
            b[ids[i], ids[i + 1]] += 1
        for i in range(n - 2):
            t[ids[i], ids[i + 1], ids[i + 2]] += 1

        p1 = u / u.sum()
        b_row = b.sum(axis=1, keepdims=True)
        p2 = np.where(b_row > 0, b / np.maximum(b_row, 1), 0.0)
        p2_s = 0.85 * p2 + 0.15 * p1[None, :]

        t_row = t.sum(axis=2, keepdims=True)
        p3 = np.where(t_row > 0, t / np.maximum(t_row, 1), 0.0)
        p3_s = 0.72 * p3 + 0.22 * p2_s[None, :, :] + 0.06 * p1[None, None, :]

        self.lp1 = np.log(np.maximum(p1, 1e-12))
        self.lp2 = np.log(np.maximum(p2_s, 1e-12))
        self.lp3 = np.log(np.maximum(p3_s, 1e-12))
        # Also store sorted letter freqs for smart init
        self.freq_order = [i for i, _ in sorted(enumerate(u), key=lambda x: -x[1])]

    def score_ids(self, arr):
        s = float(self.lp1[arr[0]] + self.lp2[arr[0], arr[1]])
        s += float(self.lp3[arr[:-2], arr[1:-1], arr[2:]].sum())
        return s

    def score_str(self, s):
        arr = np.array([self.c2i[c] for c in s if c in self.c2i], dtype=np.int32)
        return self.score_ids(arr)

def solve_masc(seq, lm, restarts=20, steps=6000, seed=42):
    rng = random.Random(seed)
    c_counts = Counter(seq)
    c_syms = [s for s, _ in c_counts.most_common()]
    k = len(c_syms)
    c2idx = {s: i for i, s in enumerate(c_syms)}
    c_list = [c2idx[s] for s in seq]
    n = len(c_list)

    V = lm.V
    if k > V:
        return -9999.0, "", {}

    # Pre-flatten tables to plain Python lists for ultra-fast O(1) indexing
    lp1 = lm.lp1.tolist()
    lp2 = lm.lp2.tolist()
    lp3_flat = lm.lp3.reshape(-1).tolist()
    V2 = V * V

    # Precompute unique trigram indices affected by each cipher symbol (0..k-1)
    # and pairs of cipher symbols -> fast incremental scoring
    sym_pos = [[] for _ in range(k)]
    for pos, ci in enumerate(c_list):
        sym_pos[ci].append(pos)

    def full_score(m):
        s = lp1[m[0]] + lp2[m[0]][m[1]]
        for t in range(n - 2):
            s += lp3_flat[(m[t] * V + m[t + 1]) * V + m[t + 2]]
        return s

    best_score = -1e18
    best_perm = None

    for r in range(restarts):
        perm = list(lm.freq_order)
        if r > 0:
            for _ in range(rng.randint(3, V)):
                i, j = rng.sample(range(V), 2)
                perm[i], perm[j] = perm[j], perm[i]
        m = [perm[ci] for ci in c_list]
        cur = full_score(m)

        t0, t1 = 4.0, 0.05
        ratio = (t1 / t0) ** (1.0 / steps)
        T = t0
        for st in range(steps):
            T *= ratio
            i = rng.randrange(k)
            j = rng.randrange(V)
            if i == j:
                continue
            pi, pj = perm[i], perm[j]
            # Collect affected text positions
            aff_p = sym_pos[i] if j >= k else (sym_pos[i] + sym_pos[j])
            # Collect unique trigram starts and header flags
            t_set = set()
            hdr0 = False
            hdr1 = False
            for p in aff_p:
                if p == 0:
                    hdr0 = True
                elif p == 1:
                    hdr1 = True
                if p >= 2:
                    t_set.add(p - 2)
                if 1 <= p <= n - 2:
                    t_set.add(p - 1)
                if p <= n - 3:
                    t_set.add(p)
            old_part = 0.0
            if hdr0:
                old_part += lp1[m[0]] + lp2[m[0]][m[1]]
            elif hdr1:
                old_part += lp2[m[0]][m[1]]
            for t in t_set:
                old_part += lp3_flat[(m[t] * V + m[t + 1]) * V + m[t + 2]]

            # Apply swap
            for p in sym_pos[i]:
                m[p] = pj
            if j < k:
                for p in sym_pos[j]:
                    m[p] = pi

            new_part = 0.0
            if hdr0:
                new_part += lp1[m[0]] + lp2[m[0]][m[1]]
            elif hdr1:
                new_part += lp2[m[0]][m[1]]
            for t in t_set:
                new_part += lp3_flat[(m[t] * V + m[t + 1]) * V + m[t + 2]]

            d = new_part - old_part
            if d >= 0.0 or rng.random() < math.exp(d / T):
                perm[i], perm[j] = pj, pi
                cur += d
            else:
                for p in sym_pos[i]:
                    m[p] = pi
                if j < k:
                    for p in sym_pos[j]:
                        m[p] = pj

        # Deterministic 2-opt polish
        improved = True
        while improved:
            improved = False
            for i in range(k):
                for j in range(i + 1, V):
                    pi, pj = perm[i], perm[j]
                    for p in sym_pos[i]:
                        m[p] = pj
                    if j < k:
                        for p in sym_pos[j]:
                            m[p] = pi
                    nxt = full_score(m)
                    if nxt > cur + 1e-9:
                        perm[i], perm[j] = pj, pi
                        cur = nxt
                        improved = True
                    else:
                        for p in sym_pos[i]:
                            m[p] = pi
                        if j < k:
                            for p in sym_pos[j]:
                                m[p] = pj

        if cur > best_score:
            best_score = cur
            best_perm = list(perm)

    mapping = {c_syms[i]: lm.i2c[best_perm[i]] for i in range(k)}
    decoded = "".join(mapping[s] for s in seq)
    return best_score, decoded, mapping

def rev_lines(seq):
    return seq[:29][::-1] + seq[29:60][::-1] + seq[60:][::-1]

# ==============================================================================
# MAIN BENCHMARK EXECUTION
# ==============================================================================
t_start = time.time()
print("=" * 82)
print("PARTIE 1 : COMPARAISON DES 5 TRANSCRIPTIONS VISUELLES + SWEEP 2^5 = 32 VARIANTES")
print("=" * 82)

print(f"{'Transcription':<16} {'Uniq':>4} {'IC':>7} {'SukV%':>6} {'VC%':>6} {'Dbl':>4} {'RevD':>5} {'Lag31':>6}")
print("-" * 62)
for name, seq in TRANSCRIPTIONS.items():
    st = seq_stats(seq)
    print(f"{name:<16} {st['uniq']:>4d} {st['ic']:>7.4f} {st['suk_v']*100:>5.1f}% {st['vc_alt']*100:>5.1f}% {st['doubles']:>4d} {st['rev_digrams']:>5d} {st['lag31']:>6d}")

# Build language models (hold out last 600 chars of raw text for clean unseen controls!)
EN_TRAIN_RAW = CORPUS_EN_FULL[:-1200]
EN_TEST_RAW = CORPUS_EN_FULL[-1200:]
FR_TRAIN_RAW = CORPUS_FR_FULL[:-900]
FR_TEST_RAW = CORPUS_FR_FULL[-900:]

REGIMES = [
    ("EN_STD",          en_std,             EN_TRAIN_RAW, EN_TEST_RAW, set("AEIOU")),
    ("EN_NO_SILENT",    en_no_silent,       EN_TRAIN_RAW, EN_TEST_RAW, set("AEIOU")),
    ("EN_PHONEMIC_24",  en_phonemic_24,     EN_TRAIN_RAW, EN_TEST_RAW, set("AEIOU")),
    ("EN_DROP_WEAK_V",  en_shorthand_drop_v,EN_TRAIN_RAW, EN_TEST_RAW, set("AEIOU")),
    ("EN_ABJAD_CONS",   en_abjad,           EN_TRAIN_RAW, EN_TEST_RAW, set("AEIOU")),
    ("FR_PHONETIC",     fr_phonetic,        FR_TRAIN_RAW, FR_TEST_RAW, set("AEIOU")),
]

models = {}
test_texts = {}
print("\n" + "=" * 82)
print("PARTIE 2A : EMPREINTE STATISTIQUE DES REGIMES PHONETIQUES / STENOGRAPHIQUES")
print("            (Comparaison vs Dorabella Hauer_2025: IC=0.0585, Uniq=20, SukV=46.0%, VC=70.9%, Dbl=4, RevD=12)")
print("=" * 82)
print(f"{'Regime':<16} {'Comp%':>6} {'|A|':>4} {'IC_cor':>7} {'IC_87(sd)':>13} {'Uniq87':>7} {'Vow%':>6} {'SukV%':>6} {'VC%':>6} {'Dbl':>5} {'RevD':>5}")
print("-" * 96)

rng_np = random.Random(123)
for rname, fn, tr_raw, te_raw, vset in REGIMES:
    raw_len = len(en_std(tr_raw))
    tr = fn(tr_raw)
    te = fn(te_raw)
    lm = LangModel(tr, rname)
    models[rname] = lm
    test_texts[rname] = te
    comp = 100.0 * len(tr) / raw_len
    ic_all = ic_of(tr)
    true_v = 100.0 * sum(1 for c in tr if c in vset) / len(tr)
    # Sample 150 slices of 87 chars from training corpus
    ics, uniqs, suks, vcs, dbls, revs = [], [], [], [], [], []
    for _ in range(150):
        pos = rng_np.randrange(0, len(tr) - 87)
        sl = list(tr[pos:pos+87])
        st = seq_stats(sl)
        ics.append(st["ic"]); uniqs.append(st["uniq"]); suks.append(st["suk_v"]*100)
        vcs.append(st["vc_alt"]*100); dbls.append(st["doubles"]); revs.append(st["rev_digrams"])
    print(f"{rname:<16} {comp:>5.1f}% {lm.V:>4d} {ic_all:>7.4f} {np.mean(ics):.4f}±{np.std(ics):.4f} {np.mean(uniqs):>6.1f} {true_v:>5.1f}% {np.mean(suks):>5.1f}% {np.mean(vcs):>5.1f}% {np.mean(dbls):>5.1f} {np.mean(revs):>5.1f}")

print("\nExemples de sorties phonetiques/stenographiques (premiers 110 symboles du corpus test):")
for rname, _, _, _, _ in REGIMES:
    print(f"  {rname:<16}: {test_texts[rname][:110]}")

# ==============================================================================
# PART 1B: STANDARD ENGLISH MASC ACROSS ALL 5 PUBLISHED TRANSCRIPTIONS + 32-FLIP SWEEP
# ==============================================================================
print("\n" + "=" * 82)
print("PARTIE 1B : TEST MASC ANGLAIS STANDARD SUR LES 5 TRANSCRIPTIONS + SWEEP 2^5=32")
print("=" * 82)
lm_en = models["EN_STD"]

# Positive control (unseen 87-char EN_STD)
ctrl_en = test_texts["EN_STD"][50:137]
# Encipher with arbitrary permutation (symbols just letters themselves — MASC is invariant)
sc_pos_en, dec_pos_en, _ = solve_masc(list(ctrl_en), lm_en, restarts=24, steps=10000, seed=101)
acc_pos_en = 100.0 * sum(a == b for a, b in zip(ctrl_en, dec_pos_en)) / 87.0
# Negative control (5 shuffles of Hauer_2025)
neg_scores_en = []
for s in range(5):
    sh = list(HAUER_2025)
    random.Random(200 + s).shuffle(sh)
    sc, _, _ = solve_masc(sh, lm_en, restarts=16, steps=8000, seed=300 + s)
    neg_scores_en.append(sc)
neg_mean_en = float(np.mean(neg_scores_en))
print(f"Calibration EN_STD (87 chars): PosCtrl={sc_pos_en:.1f} (acc={acc_pos_en:.1f}%, true_lm={lm_en.score_str(ctrl_en):.1f}) | NegCtrl_shuf={neg_mean_en:.1f} (sd={np.std(neg_scores_en):.1f})")

print(f"\n{'Transcription':<16} {'T0_Normal':>10} {'T2_RevL':>10} {'Delta_Pos':>10} {'Meilleur extrait (T0/T2)'}")
print("-" * 92)
for tname, seq in TRANSCRIPTIONS.items():
    sc0, dec0, _ = solve_masc(seq, lm_en, restarts=24, steps=10000, seed=42)
    sc2, dec2, _ = solve_masc(rev_lines(seq), lm_en, restarts=24, steps=10000, seed=43)
    best_sc, best_dec, tag = (sc0, dec0, "T0") if sc0 >= sc2 else (sc2, dec2, "T2")
    print(f"{tname:<16} {sc0:>10.1f} {sc2:>10.1f} {best_sc - sc_pos_en:>+10.1f} [{tag}] {best_dec[:55]}")

# Exhaustive 2^5 = 32 flip sweep over the 5 most visually disputed glyphs
AMBIG = [
    (12, "L1#13", "F1", "G1"),
    (21, "L1#22", "F2", "G2"),
    (33, "L2#5",  "H1", "G1"),
    (77, "L3#18", "C1", "B1"),
    (85, "L3#26", "H1", "A1"),
]
sweep_res = []
for mask in range(32):
    seq = list(HAUER_2025)
    flips = []
    for bit, (idx, lbl, orig, alt) in enumerate(AMBIG):
        if (mask >> bit) & 1:
            seq[idx] = alt
            flips.append(f"{lbl}:{alt}")
    sc, dec, _ = solve_masc(seq, lm_en, restarts=10, steps=6000, seed=500 + mask)
    sweep_res.append((sc, ic_of(seq), len(set(seq)), ",".join(flips) if flips else "NONE", dec))

sweep_res.sort(key=lambda x: -x[0])
print("\nSweep exhaustif 2^5 = 32 combinaisons sur les 5 glyphes visuellement ambigus (EN_STD T0):")
print(f"  Min={sweep_res[-1][0]:.1f} | Median={np.median([x[0] for x in sweep_res]):.1f} | Max={sweep_res[0][0]:.1f} (vs PosCtrl={sc_pos_en:.1f}, NegCtrl={neg_mean_en:.1f})")
for sc, icv, uq, fl, dec in sweep_res[:4]:
    print(f"  Top: score={sc:.1f} (IC={icv:.4f}, u={uq}) flips=[{fl}] -> {dec[:60]}")

# ==============================================================================
# PART 2B: CALIBRATED MASC ATTACK UNDER ALL 5 PHONETIC / SHORTHAND REGIMES
# ==============================================================================
print("\n" + "=" * 82)
print("PARTIE 2B : ATTAQUE MASC CALIBREE SOUS LES 5 REGIMES PHONETIQUES / STENOGRAPHIQUES")
print("            (Sur Hauer_2025, Pelling_2012, Schmeh_2018 + T2_RevL)")
print("=" * 82)

for rname, _, _, _, _ in REGIMES:
    if rname == "EN_STD":
        continue
    lm = models[rname]
    te = test_texts[rname]
    ctrl = te[20:107] if len(te) >= 107 else te[:87]
    # Check if cipher unique count exceeds alphabet
    sc_pos, dec_pos, _ = solve_masc(list(ctrl), lm, restarts=24, steps=10000, seed=600)
    acc_pos = 100.0 * sum(a == b for a, b in zip(ctrl, dec_pos)) / len(ctrl)
    true_lm = lm.score_str(ctrl)

    # Negative control (4 shuffles of Hauer_2025)
    negs = []
    for s in range(4):
        sh = list(HAUER_2025)
        random.Random(700 + s).shuffle(sh)
        sc, _, _ = solve_masc(sh, lm, restarts=14, steps=8000, seed=750 + s)
        negs.append(sc)
    neg_m = float(np.mean(negs))

    print(f"\n--- Regime: {rname} (|A|={lm.V}) ---")
    print(f"  Calibration 87-sym: True_LM={true_lm:.1f} | PosCtrl_MASC={sc_pos:.1f} (acc={acc_pos:.1f}%) | NegCtrl_shuf={neg_m:.1f}")
    print(f"    PosCtrl vrai : {ctrl[:65]}")
    print(f"    PosCtrl dec  : {dec_pos[:65]}")
    for tname in ["Hauer_2025", "Pelling_2012", "Schmeh_2018", "Visual_Alt5"]:
        seq = TRANSCRIPTIONS[tname]
        sc0, dec0, _ = solve_masc(seq, lm, restarts=24, steps=10000, seed=800)
        sc2, dec2, _ = solve_masc(rev_lines(seq), lm, restarts=24, steps=10000, seed=801)
        best_sc, best_dec, tag = (sc0, dec0, "T0") if sc0 >= sc2 else (sc2, dec2, "T2")
        print(f"  {tname:<14}: T0={sc0:>7.1f} | T2={sc2:>7.1f} | vs_Pos={best_sc - sc_pos:>+6.1f} | vs_Neg={best_sc - neg_m:>+6.1f} [{tag}] {best_dec[:52]}")

print(f"\nTemps total d'execution: {time.time() - t_start:.1f} s")
