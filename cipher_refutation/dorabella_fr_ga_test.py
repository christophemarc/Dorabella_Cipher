import math
import random
import re
import unicodedata
from collections import Counter
import numpy as np
import sys
sys.path.insert(0, '/home/user')
from dorabella_piste1_test import LangModel, clean_text, solve_masc, LINES, get_transpositions
from dorabella_piste3_test import solve_homophonic, ic_of

CORPUS_FR = """
Edward Elgar ne le deux juin mille huit cent cinquante sept a Lower Broadheath pres de Worcester et mort le vingt trois fevrier mille neuf cent trente quatre a Worcester est un compositeur et chef d orchestre britannique dont les oeuvres les plus connues sont entrees dans le repertoire classique international les Variations Enigma les marches Pomp and Circumstance son concerto pour violon son concerto pour violoncelle et deux symphonies. Il a egalement compose des oeuvres chorales dont l oratorio The Dream of Gerontius des chants et de la musique de chambre.
Bien qu Elgar soit souvent considere comme un compositeur anglais typique la plupart des musiques l ayant influence proviennent de l Europe continentale. Il se decrit comme un etranger non seulement musicalement mais egalement socialement. Dans les milieux musicaux domines par les universitaires il est reconnu comme un compositeur autodidacte. Il est aussi considere a l epoque avec mefiance en raison de sa foi catholique. Dans la classe aisee de l Angleterre victorienne et edouardienne il est gene par ses origines humbles meme lorsque son talent eut ete reconnu et qu il epousa la fille d un officier superieur de l armee britannique. Non seulement sa femme est une source d inspiration pour sa musique mais elle l introduit egalement dans les milieux aises ce qui assure sa future notoriete. Ainsi il lutte pour connaitre le succes jusqu a ses quarante ans a partir de ce moment et apres une serie de succes moderes il devient celebre au Royaume Uni et a l etranger pour ses Variations Enigma.
Dans sa cinquantaine Elgar compose une symphonie et un concerto pour violon qui rencontrent un grand succes. En revanche sa seconde symphonie et son concerto pour violoncelle attendent plusieurs annees avant d etre reconnus. La musique d Elgar dans ses dernieres annees devient attrayante pour le public britannique. Elle est peu jouee dans la periode qui suit la mort du compositeur. Mais elle revit de maniere significative dans les annees soixante aidee par de nouveaux enregistrements de ses oeuvres. Elgar est le premier compositeur a prendre le disque phonographique au serieux. Il dirige une serie d enregistrements de ses oeuvres.
Edward Elgar nait en Angleterre dans le petit village de Lower Broadheath pres de Worcester. Son pere William Henry Elgar a ete eleve a Douvres et a ete apprenti chez un editeur de musique a Londres. William demenage a Worcester ou il travaille en tant qu accordeur de piano et il ouvre un magasin de partitions et d instruments de musique. Il se marie avec Ann Greening fille d un agriculteur. Edward est le quatrieme d une fratrie de sept enfants. Ann Elgar se convertit au catholicisme peu de temps apres la naissance d Edward qui est baptise et eleve dans la foi catholique ce que son pere desapprouve. William Elgar est un violoniste de niveau professionnel et obtient le poste d organiste a l eglise catholique de Worcester. Tous les enfants Elgar recoivent une education musicale mais Edward sera toute sa vie autodidacte. A l age de huit ans Edward prend des lecons de violon et de piano. Son pere qui accorde des pianos dans de grandes maisons du Worcestershire l emmene parfois avec lui pour qu il montre ses talents aux personnalites locales.
La mere d Elgar s interesse a l art et pousse Edward a developper ses talents musicaux. Il herite d elle son discernement pour la litterature et son amour de la campagne. Son ami et biographe ecrit que l entourage d Elgar jeune a une influence sur lui qui transparait dans tout son travail et qui donne a sa vie entiere cette subtile mais neanmoins vraie et robuste qualite anglaise. Il commence a composer jeune pour une piece de theatre ecrite et jouee par les enfants Elgar alors qu il a une dizaine d annees une musique qu il arrangera et orchestrera quarante ans plus tard.
Jusqu a l age de quinze ans Elgar recoit une education generaliste pres de Worcester. Ses seules etudes musicales plus formelles en dehors des lecons de violon et de piano par des professeurs locaux sont des cours de violon donnes par Adolf Pollitzer durant de breves visites a Londres. Elgar dit la musique m a ete enseignee pour la premiere fois dans la cathedrale de Worcester par des livres que j ai empruntes a la bibliotheque musicale alors que j avais sept huit ou neuf ans. Il travaille l orgue avec des manuels et il lit tous les livres de theorie musicale qu il peut trouver. Elgar commence a apprendre l allemand avec l espoir d entrer au conservatoire de Leipzig.
Les Variations Enigma sont une oeuvre symphonique composee par le compositeur britannique Edward Elgar a base d un theme cache crypte enigmatique et de quatorze variations. Il s agit de l une de ses oeuvres les plus connues par sa musique et ses enigmes dediee a mes amis decrits ici chaque variation etant un portrait musical d un personnage de son proche entourage et finalement de lui meme.
La legende veut qu apres une journee fatigante d enseignement Elgar joue du piano. Le theme d une melodie attire alors l oreille de son epouse Caroline Alice Elgar qui lui demande de le repeter. Il lui aurait repondu oh ce n est rien mais on pourrait en faire quelque chose. A la suite de ce souhait le musicien commence a improviser des variations sur ce theme chacune d elles etant le portrait musical d un ami proche ou dans le style musical qui lui est le plus proche. Elgar en fait alors une orchestration ce qui donne ses Variations Enigma.
La piece a ete creee a Londres sous la direction de Hans Richter. Les critiques ont ete dans un premier temps quelque peu irrites par l atmosphere mystificatrice de l oeuvre mais la plupart ont loue la substance la structuration et l orchestration de la partition cette derniere devenant particulierement populaire.
L oeuvre est composee d un theme suivi par quatorze variations. Ces dernieres se basent sur la melodie du theme ainsi que sur des elements rythmiques la derniere etant sous la forme d un grand final. Sur la partition chaque variation comporte un surnom ou des initiales devant aider a l identification du portrait. Cette variation qu Elgar dedie a sa femme comporte des repetitions d une courte melodie de quatre notes qu il avait l habitude de siffler quand il revenait a son domicile. Cette variation est dediee a un ami pianiste avec qui Elgar jouait souvent de la musique de chambre. Cette variation est dediee a une amie dont le rire est decrit par les bois. Elle est de la famille de William Meath Baker decrit dans la quatrieme variation et la belle soeur de Richard Baxter Townsend decrit dans la troisieme.
"""

CORPUS_GA_RAW = """
Is oileán í Éire atá suite amuigh ó chósta iarthuaisceart mhórthír na hEorpa siar ón mBreatain Mhór in oirthear an Aigéin Atlantaigh thuaidh.
Sa mhiotaseolaíocht Ghaelach ba bhandia eapainmneach na hÉireann í Ériu iníon Ernmas de chuid na Tuatha Dé Danann. Ba é Mac Gréine a fear céile.
Tá díospóid faoi shanasaíocht Ériu ach d fhéadfadh sí a bheith díorthaithe ón bhfréamh ársa ag tagairt d uisce ag sileadh. Ós rud é go ndéanann Ériu ionadaíocht do bhandia na hÉireann go minic léirítear í mar phearsantú na hÉireann. Is í Ériu an fhoirm ársa don logainm Éire a bhíonn in úsáid sa lá atá inniu ann.
Labhraítear go minic sa Ghaeilge ar Leath Mhogha agus Leath Choinn. Clúdaíonn leath Mhogha an chuid tíre atá ó dheas ó Eiscir Riada agus Leath Choinn an chuid eile.
Tá an tír ar ar thug na Rómhánaigh Hibernia fada ó thuaidh aneas agus leathan anoir aniar. Tá maigheanna móra i lár na tíre cuid mhaith díobh clúdaithe le portaigh agus tá an chuid cnoic agus sléibhte mórthimpeall orthu ach níl mórán acu an ard. Is é Corrán Tuathail i gContae Chiarraí an ceann is airde.
Tá talamh mhaith san oirthear in áiteanna agus déantar an chuid curadóireachta mar shampla ar chruithneacht eorna biatas siúcra agus torthaí. San iarthar níl an oiread sin curadóireachta mar go bhfuil an aeráid agus an talamh tais cuid mhór den bhliain agus an talamh níos boichte i go leor áiteanna. Tógáil eallach agus caorach maraon le ba bainne an sórt talmhaíochta a chleachtaítear ansin don chuid is mó. Fanann siadsan amuigh faoin aer an bhliain ar fad de ghnáth mar níl an aimsir in Éirinn chomh crua le tíortha eile i dTuaisceart na hEorpa mar gheall ar Shruth Murascaille Mheicsiceo.
Aimsir bhog fhliuch is gnách a bheith in Éirinn. Tá tionchar mór ag Sruth Mhurascaill Mheicsiceo uirthi. Is ag an ngaoth aniar aneas atá an tionchar is mó ar an aimsir agus tig sruth na Murascaile ón treo céanna. Tugann sí aer te isteach ón gCairib a bhailíonn an chuid taiseachta ar a bealach trasna an Atlantaigh Thuaidh. Cé go bhfágann sin nach mbíonn mórán fuachta nó sneachta ann sa gheimhreadh ná teochtaí ró ard sa samhradh is minic a bhíonn an tír an ghaofar agus tais rud a bhíonn le sonrú sa gheimhreadh go háirithe.
Ós rud é go raibh an t oileán scartha amach ó mhór roinn na hEorpa de bharr ardú mara ag deireadh na hoighearaoise níl a leithéid d éagsúlacht ainmhithe is a bhfuil le fáil sa Bhreatain nó ar mhórthír na hEorpa. Na speicis a fhaightear go flúirseach ná an sionnach an ghráinneog agus an broc chomh maith leis an giorria an fia rua agus an cat crainn i líonanna níos lú. Tagtar ar roinnt ainmhithe mara i bhfarraigí an oileáin freisin mar atá an turtar mara an siorc an rón an míol mór agus an deilf. Níl nathair ar bith le fáil agus is í an laghairt an t aon reiptíl a fhaightear ann.
Tá beagnach seacht milliún duine in Éirinn agus cúig milliún sa Phoblacht. Is de bhunadh Ceilteach an chuid is mó den daonra a tháinig i dtír ó mhór roinn na hEorpa thar na cianta. Bhí cuid mhaith inimirce freisin ó am go ham ó thíortha eile go háirithe as Alba agus ar bhonn níos lú as Sasana.
Tá cáil ar na Gaeil ó thús a gcuid staire sa cheol. Bhíodh ceoltóirí agus cumadóirí ceardúla go fairsing sa tír faoi na sean ríthe agus flatha go dtí gur chuir Cath Chionn tSáile deireadh leis an seanchóras Gaelach. As sin amach leanadh den cheol ach ba cheol tíre an chuid ba mhó de le ceol rince agus oibre san áireamh. Níos déanaí tháinig foirm amhránaíochta gan tionlacan ar aghaidh ar a dtugtar an sean nós. Tá sé seo láidir fós go háirithe i nGaeltachtaí Chonnacht. Tá traidisiúin ceol fidle láidir i gcuid mhór den tír ach castar ceol traidisiúnta freisin ar an gcruit ar an bhfeadóg stáin agus ar an mbosca ceoil go forleathan. Bhíodh rincí traidisiúnta go rialta ag na crosbhóthair ach ina dhiaidh sin ba sna hallaí rince ba mhó a bhí ceol le haireachtáil go beo.
Maraon leis an gceol tá cáil ar na Gaeil leis an scríbhneoireacht. San fhichiú céad is i mBéarla atá an chuid is mó den scríbhneoireacht seo ar aon dul leis an titim ar úsáid na Gaeilge thar an achar sin. Tá an chuid is mó de na meáin i bPoblacht na hÉireann bunaithe i mBaile Átha Cliath.
Le linn tréimhse na hoighearaoise deireanaí clúdaíodh oighir go tréimhsiúil formhór na hÉireann. Bhí leibhéil na farraige níos ísle agus bhí Éire cosúil leis an mBreatain Mhór mar chuid de mhór roinn na hEorpa. Ba chúis le hardú na farraige de bharr leá oighir Éire a bheith scartha ón mBreatain Mhór.
Is ceann de na teangacha Ceilteacha í an Ghaeilge nó Gaeilge na hÉireann mar a thugtar uirthi corruair agus ceann de na trí cinn de theangacha Ceilteacha ar a dtugtar na teangacha Gaelacha go háirithe. Labhraítear in Éirinn go príomha í ach tá cainteoirí Gaeilge ina gcónaí in áiteanna eile ar fud an domhain. Is í an teanga náisiúnta nó dhúchais agus an phríomhtheanga oifigiúil i bPoblacht na hÉireann í an Ghaeilge. Tá an Béarla luaite sa Bhunreacht mar theanga oifigiúil eile. Tá aitheantas oifigiúil aici chomh maith i dTuaisceart Éireann.
Ba é an tOgham an chéad chóras scríofa don Ghaeilge ach is gnách an teanga a chur ar pár i litreacha Laidineacha le níos mó ná míle bliain anuas. Sa tseachtú haois a tháinig an chéad leagan caighdeánaithe den teanga mar atá an tSean Ghaeilge ar an bhfód agus í sách difriúil fós leis an gcineál Gaeilge a labhraítear inniu. De réir a chéile áfach thosaigh na manaigh ag breacadh síos na scéalta miotaseolaíochta. De réir a chéile tháinig meath ar thraidisiún scríofa na Sean Ghaeilge agus an Mheán Ghaeilge ag teacht ina háit. Teanga cineál measctha is ea an Mheán Ghaeilge nó is deacair aon chaighdeán nó norm ceart a leagan amach di.
"""

# Also build a 23-symbol Irish LangModel preserving fada vowels: á->J, é->K, í->Q, ó->V, ú->W!
# (Since J, K, Q, V, W are absent from native Irish!)
def clean_irish_fada(txt):
    rep = {
        "á": "J", "Á": "J",
        "é": "K", "É": "K",
        "í": "Q", "Í": "Q",
        "ó": "V", "Ó": "V",
        "ú": "W", "Ú": "W",
    }
    out = []
    for c in txt:
        if c in rep:
            out.append(rep[c])
        else:
            c2 = unicodedata.normalize("NFKD", c)
            c2 = "".join(ch for ch in c2 if not unicodedata.combining(ch)).upper()
            if "A" <= c2 <= "Z":
                out.append(c2)
    return "".join(out)

def pretty_fada(s):
    return s.replace("J", "Á").replace("K", "É").replace("Q", "Í").replace("V", "Ó").replace("W", "Ú")

class FadaIrishLM(LangModel):
    def __init__(self, raw_text):
        self.name = "Gaeilge (23 signes avec fada)"
        s = clean_irish_fada(raw_text)
        self.corpus_len = len(s)
        arr = [ord(c) - 65 for c in s]
        c1 = np.zeros(26, dtype=np.float64)
        c2 = np.zeros((26, 26), dtype=np.float64)
        c3 = np.zeros((26, 26, 26), dtype=np.float64)
        for a in arr: c1[a] += 1
        for a, b in zip(arr, arr[1:]): c2[a, b] += 1
        for a, b, c in zip(arr, arr[1:], arr[2:]): c3[a, b, c] += 1
        p1 = (c1 + 0.5) / (c1.sum() + 13.0)
        p2 = (c2 + 0.2 * p1[None, :]) / (c2.sum(axis=1, keepdims=True) + 0.2)
        p3 = (c3 + 0.5 * p2[None, :, :]) / (c3.sum(axis=2, keepdims=True) + 0.5)
        self.logp3 = np.log(0.72 * p3 + 0.23 * p2[None, :, :] + 0.05 * p1[None, None, :])
        self.logp2 = np.log(0.85 * p2 + 0.15 * p1[None, :])
        self.logp1 = np.log(p1)
        self.freq_order = list(np.argsort(-c1))

# Controls (87 letters, NOT in training text!)
CTRL_FR = clean_text("Chere Mademoiselle Dora nous vous remercions vivement de votre accueil si chaleureux en juillet et esperons vous revoir tres bientot")[:87]
CTRL_GA = clean_text("A chara dhilis Dora go raibh mile maith agat as na laethanta brea a chaitheamar le cheile i do theach i mi Iuil agus ta suil agam go mbuailfimid le cheile go luath")[:87]
CTRL_GA_FADA = clean_irish_fada("A chara dhílis Dora go raibh míle maith agat as na laethanta breátha a chaitheamar le chéile i do theach i mí Iúil agus tá súil agam go mbuailfimid le chéile go luath")[:87]

def main():
    lm_fr = LangModel("Français", CORPUS_FR)
    lm_ga = LangModel("Gaeilge (18 lettres)", CORPUS_GA_RAW)
    lm_ga_f = FadaIrishLM(CORPUS_GA_RAW)

    # Corpus statistics: natural IC and vowel ratio!
    fr_clean = clean_text(CORPUS_FR)
    ga_clean = clean_text(CORPUS_GA_RAW)
    gaf_clean = clean_irish_fada(CORPUS_GA_RAW)

    print("=== STATISTIQUES NATURELLES DES CORPORA vs DORABELLA (IC=0.0585, 20 uniques) ===")
    for name, s in [("Français (FR)", fr_clean), ("Irlandais 18L (GA)", ga_clean), ("Irlandais 23L+fada (GA_F)", gaf_clean)]:
        v_pct = 100 * sum(1 for c in s if c in "AEIOUYJKQVW") / len(s)
        print(f"  {name:26s} | N={len(s):5d} | IC naturel={ic_of(s):.4f} | uniques/87car~{len(set(s[:87]))} | voyelles={v_pct:.1f}%")

    rng = random.Random(2026)
    base_dora = LINES[0] + LINES[1] + LINES[2]
    transps = get_transpositions()
    test_t_keys = [
        "T0_Normal (L1->L3, G->D)",
        "T1_Inverse_180 (L3->L1, D->G)",
        "T2_Lignes_inversees (L1->L3, D->G)",
        "T3_Boustrophedon_A (->, <-, ->)",
        "T5_Colonnes_gauche (haut->bas)",
        "T8_Lignes_bas_haut (L3,L2,L1 G->D)"
    ]

    configs = [
        ("FR", lm_fr, CTRL_FR, False),
        ("GA_18", lm_ga, CTRL_GA, False),
        ("GA_23_FADA", lm_ga_f, CTRL_GA_FADA, True),
    ]

    for tag, lm, pt_ctrl, is_fada in configs:
        print("\n" + "=" * 76)
        print(f"LANGUE : {tag} ({lm.name})")
        print("=" * 76)
        # Positive control
        alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        sh = alpha[:]
        rng.shuffle(sh)
        enc = dict(zip(alpha, sh))
        ct_pos = [enc[c] for c in pt_ctrl]
        sc_pos, dec_pos = solve_masc(ct_pos, lm, restarts=12, steps=6000, seed=42)
        acc = sum(1 for a, b in zip(pt_ctrl, dec_pos) if a == b) / 87.0

        # Negative control (3 shuffles of Dorabella)
        neg = []
        for s_i in range(3):
            sd = base_dora[:]
            rng.shuffle(sd)
            sn, _ = solve_masc(sd, lm, restarts=8, steps=5000, seed=100+s_i)
            neg.append(sn)
        mean_neg, max_neg = sum(neg)/len(neg), max(neg)

        disp_orig = pretty_fada(pt_ctrl) if is_fada else pt_ctrl
        disp_dec  = pretty_fada(dec_pos) if is_fada else dec_pos
        print(f"[CALIBRATION] Vrai texte 87 car. (score={sc_pos:.1f}, exactitude={100*acc:.1f}%) | Hasard moy={mean_neg:.1f} (max={max_neg:.1f})")
        print(f"  Original : {disp_orig}")
        print(f"  Solveur  : {disp_dec}")

        print(f"\n[ATTAQUE MASC — 6 TRANSPOSITIONS]")
        best_t = (-1e18, "", "")
        for tk in test_t_keys:
            sc, dec = solve_masc(transps[tk], lm, restarts=12, steps=6000, seed=777)
            gap = sc - mean_neg
            disp = pretty_fada(dec) if is_fada else dec
            print(f"  {tk:35s} | score={sc:6.1f} (vs hasard {gap:+5.1f}) | {disp[:52]}...")
            if sc > best_t[0]:
                best_t = (sc, tk, disp)
        print(f"  -> Meilleur MASC ({best_t[1]}): {best_t[2]}")

        # Also quick Homophonic check (k=3, 4 merges on T0 Normal)
        print(f"\n[ATTAQUE HOMOPHONIQUE (T0 Normal, k=3 et k=4 fusions)]")
        sd = base_dora[:]
        rng.shuffle(sd)
        o_neg, _, _, _ = solve_homophonic(sd, lm, max_merges=4, restarts=6, steps=4000, seed=88)
        for km in (3, 4):
            o_d, r_d, dec_h, hom_d = solve_homophonic(base_dora, lm, max_merges=km, restarts=12, steps=5500, seed=90+km)
            disp_h = pretty_fada(dec_h) if is_fada else dec_h
            print(f"  k={km} fusions | obj={o_d:6.1f} (raw={r_d:6.1f}) vs Hasard(k=4)={o_neg:.1f} | homos={hom_d}")
            print(f"     -> {disp_h}")

if __name__ == "__main__":
    main()
