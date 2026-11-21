import math
import random
import re
import unicodedata
from collections import Counter
import numpy as np

# ==============================================================================
# 1. AUTHENTIC CORPORA (EN, IT, DE) — cleaned prose from Wikipedia articles
# ==============================================================================
CORPUS_EN = """
Sir Edward William Elgar was an English composer many of whose works have entered the British and international classical concert repertoire. Among his best known compositions are orchestral works including the Enigma Variations the Pomp and Circumstance Marches concertos for violin and cello and two symphonies. He also composed choral works including The Dream of Gerontius chamber music and songs. He was appointed Master of the Kings Musick in nineteen twenty four.
Although Elgar is often regarded as a typically English composer most of his musical influences were not from England but from continental Europe. He felt himself to be an outsider not only musically but socially. In musical circles dominated by academics he was a self taught composer in Protestant Britain his Roman Catholicism was regarded with suspicion in some quarters and in the class conscious society of Victorian and Edwardian Britain he was acutely sensitive about his humble origins even after he achieved recognition. He nevertheless married the daughter of a senior British Army officer. She inspired him both musically and socially but he struggled to achieve success until his forties when after a series of moderately successful works his Enigma Variations became immediately popular in Britain and overseas. He followed the Variations with a choral work The Dream of Gerontius based on a Roman Catholic text that caused some disquiet in the Anglican establishment in Britain but it became and has remained a core repertory work in Britain and elsewhere.
In his fifties Elgar composed a symphony and a violin concerto that were immensely successful. His second symphony and his cello concerto did not gain immediate public popularity and took many years to achieve a regular place in the concert repertory of British orchestras. Elgars music came in his later years to be seen as appealing chiefly to British audiences. His stock remained low for a generation after his death. It began to revive significantly in the nineteen sixties helped by new recordings of his works.
Edward Elgar was born in the small village of Lower Broadheath near Worcester England. His father William Henry Elgar was raised in Dover and had been apprenticed to a London music publisher. William moved to Worcester where he worked as a piano tuner and set up a shop selling sheet music and musical instruments. He married Ann Greening daughter of a farm worker. Edward was the fourth of their seven children. Ann Elgar had converted to Roman Catholicism shortly before Edwards birth and he was baptised and brought up as a Roman Catholic to the disapproval of his father. William Elgar was a violinist of professional standard and held the post of organist of Saint Georges Roman Catholic Church Worcester. All the Elgar children received a musical upbringing. By the age of eight Elgar was taking piano and violin lessons and his father who tuned the pianos at many grand houses in Worcestershire would sometimes take him along giving him the chance to display his skill to important local figures.
Elgars mother was interested in the arts and encouraged his musical development. He inherited from her a discerning taste for literature and a passionate love of the countryside. His friend and biographer wrote that Elgars early surroundings had an influence that permeated all his work and gave to his whole life that subtle but none the less true and sturdy English quality. He began composing at an early age for a play written and acted by the Elgar children when he was about ten he wrote music that forty years later he rearranged with only minor changes and orchestrated as the suites titled The Wand of Youth.
Until he was fifteen Elgar received a general education at Littleton House school near Worcester. His only formal musical training beyond piano and violin lessons from local teachers consisted of more advanced violin studies during brief visits to London. Elgar said my first music was learnt in the Cathedral from books borrowed from the music library when I was eight nine or ten. He worked through manuals of instruction on organ playing and read every book he could find on the theory of music. Elgar began to learn German in the hope of going to the Leipzig Conservatory for further musical studies but his father could not afford to send him. However it was a disappointment to Elgar that on leaving school he went not to Leipzig but to the office of a local solicitor as a clerk. He did not find an office career congenial and for fulfilment he turned not only to music but to literature becoming a voracious reader.
After a few months Elgar left the solicitor to embark on a musical career giving piano and violin lessons and working occasionally in his fathers shop. He was an active member of the Worcester Glee club along with his father and he accompanied singers played the violin composed and arranged works and conducted for the first time. Although rather solitary and introspective by nature Elgar thrived in Worcesters musical circles. He played in the violins at the Worcester and Birmingham Festivals and regularly played the bassoon in a wind quintet alongside his brother Frank an oboist.
Jane Austen was an English writer known primarily for her six novels which implicitly interpret critique and comment on the English landed gentry at the end of the eighteenth century. Austens plots often explore the dependence of women on marriage for the pursuit of favourable social standing and economic security. Her works are implicit critiques of the novels of sensibility of the second half of the eighteenth century and are part of the transition to nineteenth century literary realism. Her use of social commentary realism wit and irony have earned her acclaim amongst critics and scholars. Austen wrote major novels before the age of twenty two but she was not published until she was thirty five. The anonymously published Sense and Sensibility Pride and Prejudice Mansfield Park and Emma were moderate successes but they did not bring her public fame in her lifetime. She wrote two other novels Northanger Abbey and Persuasion both published posthumously and began another eventually titled Sanditon but it was left unfinished on her death. Since her death Austens novels have rarely been out of print. A significant transition in her reputation occurred when they were republished in Standard Novels series. They gradually gained wide acclaim and popular readership.
The scant biographical information about Austen comes from her surviving letters and a number of sketches her family members and contemporaries wrote about her. Only about one hundred and sixty of the approximately three thousand letters Austen may have written in the course of her lifetime have survived and been published. It is believed that Cassandra Austen destroyed the bulk of the letters she received from her sister. One theory is that she wanted to ensure that the younger nieces did not read any of Janes sometimes acid or forthright comments on neighbours or family members. In the interest of protecting reputations from Janes penchant for honesty and forthrightness Cassandra it is surmised may have omitted details of illnesses unhappiness and anything she considered unsavoury. Austens family and relatives built a legend of good quiet Aunt Jane portraying her as a woman in a happy domestic situation whose family was the mainstay of her life. Critics have long taken issue with this depiction of a mild Austen. George served as the rector of the Anglican parishes of Steventon and Deane. The Reverend Austen came from an old and wealthy family of wool merchants. As each generation of eldest sons received inheritances Georges branch of the family fell into poverty. He and his two sisters were orphaned as children and had to be taken in by relatives.
My dear friend I hope you are very well and that the summer weather in Wolverhampton continues to be pleasant after our delightful visit last week. We returned home to Malvern yesterday evening and found the garden looking fresh after the rain. Please give my warmest regards to your father and to all the family and tell them how much Alice and I enjoyed our stay with you. I have been working on a new piece of music since Monday morning and hope to play it for you when next we meet.
"""

CORPUS_IT = """
Cosi fan tutte ossia La scuola degli amanti e un opera lirica in due atti di Wolfgang Amadeus Mozart scritta fra il mille settecento ottantanove e il mille settecento novanta. E la terza e ultima delle tre opere buffe scritte dal compositore salisburghese su libretto di Lorenzo da Ponte.
Nonostante fosse stato nominato compositore di corte e ricevesse per l incarico uno stipendio di novecento fiorini all anno ragguardevole per l epoca la somma non era sufficiente per le necessita familiari di Mozart che era sempre in difficolta economiche. Chiese spesso prestiti ad amici e scrisse diversi lavori cameristici con lo scopo di venderli. Quando durante l estate la moglie Constanze si ammalo le cure a cui dovette sottoporsi alla stazione termale di Baden aggravarono ulteriormente la situazione finanziaria familiare. La commissione per il teatro di corte da parte dell imperatore Giuseppe secondo in seguito alle felici riprese viennesi di Le nozze di Figaro e Don Giovanni sembro provvidenziale al musicista.
Sembra che a suggerire l argomento dell opera a Da Ponte sia stato proprio il committente l imperatore sulla base di fatti realmente accaduti ma secondo molti studiosi tale dettaglio e stato ideato per giustificare un testo ben poco apprezzato avvalorandolo con il presunto suggerimento imperiale. Avuto l incarico Mozart collaboro nuovamente con Lorenzo da Ponte che prese alcune idee per la nuova opera da Le metamorfosi di Ovidio e da La grotta di Trofonio con altri riferimenti a Goldoni Marivaux e Ariosto anche se la vicenda e in gran parte originale del librettista. Mozart scrisse la partitura tra l ottobre e il gennaio probabilmente l opera era gia quasi completa alla fine di dicembre poiche il musicista ne realizzo un esecuzione privata nella sua abitazione la sera di San Silvestro alla presenza di diversi amici e musicisti tra cui Haydn. La prima rappresentazione ebbe luogo al Burgtheater di Vienna e fu diretta dal compositore. Il successo fu discreto ma l opera rimase in scena per sole quattro sere sospesa bruscamente alla morte dell imperatore.
Il pubblico viennese accolse tiepidamente Cosi fan tutte probabilmente turbato dall argomento al limite della moralita per di piu la presenza di doppi sensi e di paragoni impudenti sconcerto gli ascoltatori appartenenti ormai a un epoca vicina alla sensibilita romantica per tutto il diciannovesimo secolo infatti l opera non venne piu rappresentata. Il giudizio sulla composizione fu per molto tempo nell Ottocento unanime musica molto bella ma libretto improponibile.
In una bottega di caffe a Napoli siedono i due ufficiali dell esercito Ferrando e Guglielmo che vantano la fedelta delle loro fidanzate rispettivamente Dorabella e Fiordiligi sorelle giunte da Ferrara. Con loro vi e l amico Don Alfonso scapolo il quale dandosi come sempre arie da filosofo cinico li contraddice affermando l inesistenza della fedelta femminile e sostenendo cosi che le due innamorate se solo si presentasse loro l occasione dimenticherebbero subito i fidanzati e passerebbero a nuovi amori. A seguito di questa dichiarazione i due intendono sfidare a duello l amico per difendere l onore delle future spose. Don Alfonso invece lancia loro una scommessa dell ingente valore di cento zecchini per provare ai due amici che le loro fidanzate non son diverse dalle altre donne cosi per un giorno Ferrando e Guglielmo dovranno comportarsi secondo gli ordini di Don Alfonso.
Nel frattempo nel giardino della loro casa sul Golfo di Napoli Fiordiligi e Dorabella guardano sognanti i ritratti dei fidanzati ma poi iniziano a preoccuparsi perche sono gia le sei del pomeriggio e i due uomini non sono ancora venuti a far loro visita come fanno di solito tutti i giorni. Ad arrivare e invece Don Alfonso che reca loro una notizia terribile i fidanzati sono stati convocati al fronte e devono partire all istante. Sopraggiungono Ferrando e Guglielmo che fingono di essere in partenza. Le due giovani sono disperate e gli addii sono dolorosi. Don Alfonso osserva con allegro cinismo che sanno tutti due far ben la loro parte. Un suono di tamburo richiama gli ufficiali al loro dovere e l imbarcazione delle reclute li accoglie.
La cameriera Despina complice di Don Alfonso espone alle sorelle le proprie idee circa la fedelta maschile ed esorta Fiordiligi e Dorabella a far all amor come assassine sostenendo che sicuramente i fidanzati al fronte faranno altrettanto. Don Alfonso cerca l aiuto di Despina offrendole subito uno zecchino d oro e promettendole venti scudi se insieme riusciranno a far entrare nelle grazie delle sorelle due nuovi pretendenti. Questi ultimi non sono altro che gli stessi Ferrando e Guglielmo che si ripresentano al loro cospetto travestiti da ufficiali albanesi con grandi mustacchi e vestiti stravaganti. Le due giovani irrompono furenti per la presenza degli sconosciuti rimproverando Despina per averli fatti entrare i finti albanesi si dichiarano spasimanti delle due sorelle e Don Alfonso presenta gli ufficiali come suoi cari amici. Alle loro rinnovate e caricaturali profferte d amore Fiordiligi risponde che serberanno fedelta agli amanti fino alla morte e quindi con la sorella si ritira. Tutto cio rallegra i due uomini che sentono gia di aver vinto la scommessa ma Don Alfonso li ammonisce avvertendoli che c e ancora tempo prima di cantare vittoria.
Vedendo che le giovani se ne stanno malinconiche in giardino Despina confida al filosofo di avere un piano per cambiare la situazione. Don Alfonso si allontana fingendo di inseguire gli albanesi che poco lontano sotto gli occhi delle due fanciulle simulano di bere dell arsenico volendosi suicidare per il dolore. Don Alfonso fa finta di andare in cerca di un medico e lascia i due uomini agonizzanti davanti alle esterrefatte sorelle che iniziano a provare compassione. Arriva Despina travestita da medico declamando frasi in un latino maccheronico e fa rinvenire gli albanesi toccandoli con una calamita.
Nella loro camera Fiordiligi e Dorabella vengono convinte da Despina a divertirsi un poco e non morire dalla malinconia senza mancare di fede agli amanti. Nel giardino sul mare i due albanesi hanno organizzato una serenata alle dame con i suonatori e i cantanti che arrivano in barca. Don Alfonso e Despina incoraggiano gli amanti e le donne a parlarsi e li lasciano soli. Fiordiligi e Ferrando si allontanano suscitando la gelosia di Guglielmo che offre un regalo a Dorabella e riesce a conquistarla. Fiordiligi e sconvolta perche sta capendo che il gioco si e mutato in realta. Quando Ferrando si accomiata Fiordiligi ha un attimo di debolezza e vorrebbe richiamarlo poi rivolge il pensiero al promesso sposo Guglielmo e si proclama a lui fedele.
Giuseppe Fortunino Francesco Verdi e stato un compositore drammaturgo e politico italiano. Universalmente riconosciuto come uno dei maggiori operisti d ogni tempo le sue opere rimangono ancora tra le piu conosciute ed eseguite nei teatri internazionali in particolare il Nabucco la cosiddetta Trilogia popolare composta da Rigoletto Il trovatore e La traviata e Aida. Nel corso della sua lunga vita stabili una posizione unica tra i suoi connazionali divenendo un simbolo artistico profondo dell unita politica e culturale del Paese. Giuseppe Verdi nacque a Roncole di Busseto nel Ducato di Parma e Piacenza da genitori piacentini Carlo oste e rivenditore di sale e generi alimentari e Luigia Uttini filatrice operaia. L eccezionale talento compositivo di Verdi fu indubbiamente coltivato e accresciuto dallo studio ma fu anche sostenuto dal padre che intuiva le ottime prospettive del figlio. Anche Pietro Baistrocchi organista della chiesa di paese che prese il ragazzo a benvolere lo avvio gratuitamente alla pratica dell organo e del pianoforte. Piu tardi Antonio Barezzi un negoziante amante della musica e direttore della locale societa filarmonica convinto che la fiducia nel giovane fosse ben riposta divenne suo mecenate e protettore aiutandolo a proseguire gli studi.
Nel giugno Verdi si diplomo presso il Ginnasio e pote cosi dedicarsi esclusivamente alla musica sotto la guida di Provesi. All eta di tredici anni gli fu chiesto di sostituire un musicista in quello che divenne il suo primo evento pubblico nella citta natale riscuotendo un grande successo.
"""

CORPUS_DE = """
Sir Edward William Elgar war ein britischer Komponist. Sein bekanntestes Werk ist der Pomp and Circumstance March. Der Mittelteil ist auch als Hymne Land of Hope and Glory weltbekannt und gilt als die wichtigste inoffizielle Hymne des Vereinigten Koenigreichs das Stueck ist ein Hoehepunkt der alljaehrlichen Last Night of the Proms.
Edward Elgar war der Sohn des Musikalienhaendlers Klavierstimmers und Organisten William Henry Elgar und verbrachte seine ersten Jahre ueber der elterlichen Musikalienhandlung. Er war das vierte von sieben Kindern. Seine Mutter Ann Elgar geborene Greening war gegen den Willen ihres Vaters zum katholischen Glauben konvertiert und erzog ihre Kinder katholisch. Elgar spielte bereits frueh verschiedene Instrumente. Nach kurzer Arbeit bei einem Notar stieg er in das vaeterliche Geschaeft ein. Mit sechzehn Jahren beschloss er eine musikalische Ausbildung zu absolvieren. Er leitete die Proben und begann Fagott zu spielen. Er stellte ein atypisches Blaeserquintett mit den vorhandenen Musikern auf bestehend aus zwei Floeten Oboe Klarinette und Fagott wobei er haeufig das Fagott spielte. Die Kammermusik fuer diese Besetzung schrieb er selbst einfache und gefaellige Gebrauchsmusik zur sonntaeglichen Auffuehrung.
In den fruehen Tagen von denen ich sprach setzten sich fuenf von uns zu einem Blaeserquintett zusammen. Wir hatten zwei Floeten eine Oboe eine Klarinette und ein Fagott. Letzteres spielte ich einige Zeit selbst bis ich es spaeter zugunsten des Cellos aufgab. Es gab ueberhaupt keine Musik die zu unseren eigenartigen Beduerfnissen gepasst haette da in einem idealen Blaeserquintett ein Horn zu finden sein sollte und keine zweite Floete deshalb schrieb ich meist die Musik. Wir trafen uns an den Sonntagsnachmittagen und es war eine selbstverstaendliche Sache dass wir jede Woche ein neues Stueck hatten. Die Predigten in unserer Kirche dauerten ueblicherweise mindestens eine halbe Stunde und ich verbrachte die Zeit damit das Stueck fuer den Nachmittag zu komponieren. Es war eine grosse Erfahrung fuer mich wie Sie sich vorstellen koennen und da die Buecher noch alle vorhanden sind gibt es einige dieser Musikstuecke noch immer. Gelegentlich spielten wir fuer Freunde und ich erinnere mich an eine Mondnacht in der ich vor einem Haus stand um das Fagott zusammenzusetzen.
Elgar wurde in Worcester Konzertmeister und als Nachfolger seines Vaters Organist an der Kirche Saint George. Er heiratete seine Klavierschuelerin Caroline Alice Roberts zog nach Malvern in Worcestershire und lebte danach als freischaffender Komponist. Sie unterstuetzte ihren Mann bei dessen Arbeit und fungierte als seine Managerin. Das Paar bekam eine Tochter Carice. Seine ersten Kantaten sowie das Oratorium verschafften ihm Anerkennung doch der endgueltige Durchbruch als Komponist gelang Elgar mit seinen Enigma Variationen und ein Jahr spaeter mit dem Oratorium fuer das Birmingham Festival. Die Enigma Variationen ein grossformatiges Orchesterwerk mit vierzehn Charaktervariationen symbolisieren in kunstvoll farbiger Instrumentation jeweils eine persoenliche Eigenschaft eines Freundes beziehungsweise einer Freundin des Komponisten und knuepfen an die grosse Tradition der Orchestervariation in der Musik des neunzehnten Jahrhunderts an.
In den Jahren vor und waehrend des Ersten Weltkriegs schuf Elgar neben patriotischen Stuecken drei Kammermusikwerke das elegische Cellokonzert und das fertiggestellte Konzert fuer Violine und Orchester das er auf Wunsch von Fritz Kreisler komponierte und in die Tradition gross angelegter romantischer Konzerte wie die von Beethoven und Brahms stellte. Es wurde ein grosser Publikumserfolg und wird bis heute von fuehrenden Violinisten eingespielt. Mit einer Laenge von rund fuenfzig Minuten gehoert es zu den laengsten Instrumentalkonzerten der Musikgeschichte. Besonders erfolgreich war die erste Sinfonie waehrend die zweite Sinfonie weniger beifaellig aufgenommen wurde was moeglicherweise an dem introspektiven Finale liegt.
Der Tod seiner Frau Caroline Alice liess Elgars Schaffenskraft versiegen. Hinzu kam dass sein Kompositionsstil in den zwanziger Jahren als altmodisch angesehen wurde. Nach dem Tod seiner Frau zog sich Elgar nach Worcestershire zurueck um mit seinen Hunden das Leben eines Landedelmanns zu fuehren. Auch interessierte er sich fuer Chemie und richtete sich in seinem Haus in Hereford ein eigenes Laboratorium ein. Als Symphoniker verband er kosmopolitische Attituede mit englischer Melodieverwurzelung und unterscheidet sich nicht zuletzt durch seine Instrumentierungskunst von seinen Zeitgenossen. Doch zeigen seine besten Werke einen Tonsetzer von bemerkenswertem musikalischem Empfindungsreichtum.
Johann Wolfgang von Goethe war ein deutscher Dichter Politiker und Naturforscher. Er gilt als einer der bedeutendsten Schoepfer deutschsprachiger Dichtung. Goethe stammte aus einer wohlhabenden und angesehenen buergerlichen Familie sein Grossvater muetterlicherseits war hoechster Justizbeamter der Stadt Frankfurt am Main sein Vater Doktor der Rechte und Kaiserlicher Rat. Er und seine Schwester Cornelia erfuhren eine aufwendige Ausbildung durch Hauslehrer. Dem Wunsch seines Vaters folgend studierte Goethe in Leipzig und Strassburg Rechtswissenschaft und war danach als Advokat in Wetzlar und Frankfurt taetig. Gleichzeitig folgte er seiner Neigung zur Dichtkunst. Die ersten Anerkennungen in der Welt der Literatur erzielte er mit dem Drama Goetz von Berlichingen das ihm nationalen Erfolg eintrug und mit dem Briefroman Die Leiden des jungen Werthers mit dem er sogar europaeischen Erfolg erreichte.
Der jugendliche Herzog Carl August von Sachsen Weimar Eisenach berief den jungen Goethe an seinen Hof in Weimar wo er schliesslich fuer den Rest seines Lebens blieb. Er bekleidete dort als Freund und Minister des Herzogs politische und administrative Aemter und leitete ein Vierteljahrhundert das Weimarer Hoftheater. Die aus seiner amtlichen Taetigkeit resultierende Vernachlaessigung seiner schoepferischen Faehigkeiten loeste nach dem ersten Weimarer Jahrzehnt eine persoenliche Krise aus der sich Goethe durch die Flucht nach Italien entzog. Die Italienreise empfand er als eine Wiedergeburt. Nach seiner Rueckkehr wurden seine Amtspflichten weitgehend auf repraesentative Aufgaben beschraenkt. Der in Italien erlebte Reichtum an kulturellem Erbe stimulierte seine dichterische Produktion. Goethes literarisches Werk umfasst Lyrik Dramen Epik autobiografische kunst und literaturtheoretische sowie naturwissenschaftliche Schriften. Daneben ist sein umfangreicher Briefwechsel von literarischer Bedeutung. Sein Drama Faust errang den Ruf als die bedeutendste Schoepfung der deutschsprachigen Literatur.
Johann Wolfgang von Goethe wurde in Frankfurt am Main geboren und tags darauf evangelisch getauft. Sein Rufname war Wolfgang. Auch als Kind und Jugendlicher war Goethe haeufig krank und litt insbesondere schwer an einer Pockenerkrankung. Goethes Vater hatte zwar an der Leipziger Universitaet den Doktorgrad der Jurisprudenz erworben uebte aber keinen juristischen Beruf aus. Er war vielseitig interessiert und gebildet jedoch auch streng und pedantisch was wiederholt zu Konflikten in der Familie fuehrte. Die Geschwister erhielten eine aufwendige Ausbildung. Danach wurden er und seine Schwester gemeinsam vom Vater sowie von insgesamt acht Hauslehrern unterrichtet. Goethe erlernte Latein Griechisch und Hebraeisch als klassische Bildungssprachen sowie die lebenden Sprachen Franzoesisch Italienisch und Englisch.
"""

def clean_text(txt):
    txt = txt.replace("ß", "SS").replace("ä", "AE").replace("ö", "OE").replace("ü", "UE")
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"[^A-Z]", "", txt.upper())

# ==============================================================================
# 2. LANGUAGE MODEL (Interpolated Unigram + Bigram + Trigram log-prob)
# ==============================================================================
class LangModel:
    def __init__(self, name, raw_text):
        self.name = name
        s = clean_text(raw_text)
        self.corpus_len = len(s)
        arr = [ord(c) - 65 for c in s]
        
        c1 = np.zeros(26, dtype=np.float64)
        c2 = np.zeros((26, 26), dtype=np.float64)
        c3 = np.zeros((26, 26, 26), dtype=np.float64)
        
        for a in arr:
            c1[a] += 1
        for a, b in zip(arr, arr[1:]):
            c2[a, b] += 1
        for a, b, c in zip(arr, arr[1:], arr[2:]):
            c3[a, b, c] += 1
            
        p1 = (c1 + 0.5) / (c1.sum() + 13.0)
        p2 = (c2 + 0.2 * p1[None, :]) / (c2.sum(axis=1, keepdims=True) + 0.2)
        p3 = (c3 + 0.5 * p2[None, :, :]) / (c3.sum(axis=2, keepdims=True) + 0.5)
        
        p_interp = 0.72 * p3 + 0.23 * p2[None, :, :] + 0.05 * p1[None, None, :]
        self.logp3 = np.log(p_interp)
        self.logp2 = np.log(0.85 * p2 + 0.15 * p1[None, :])
        self.logp1 = np.log(p1)
        self.freq_order = list(np.argsort(-c1))

    def score_text(self, txt):
        seq = [ord(c) - 65 for c in txt]
        sc = self.logp1[seq[0]] + self.logp2[seq[0], seq[1]]
        lp3 = self.logp3
        for i in range(len(seq) - 2):
            sc += lp3[seq[i], seq[i+1], seq[i+2]]
        return float(sc)

# ==============================================================================
# 3. SIMULATED ANNEALING MASC SOLVER (1-to-1 bijection over 26 letters)
# ==============================================================================
def solve_masc(cipher_tokens, lm, restarts=12, steps=6000, seed=1897):
    rng = random.Random(seed)
    counts = Counter(cipher_tokens)
    uniq = [u for u, _ in counts.most_common()]
    sym2id = {u: i for i, u in enumerate(uniq)}
    c_ids = [sym2id[t] for t in cipher_tokens]
    n_uniq = len(uniq)
    n_len = len(c_ids)
    lp1, lp2, lp3 = lm.logp1, lm.logp2, lm.logp3
    
    best_overall_score = -1e18
    best_overall_map = None

    for r in range(restarts):
        perm = lm.freq_order[:]
        if r > 0:
            for _ in range(rng.randint(4, 18)):
                i, j = rng.randrange(26), rng.randrange(26)
                perm[i], perm[j] = perm[j], perm[i]
        
        seq = [perm[c] for c in c_ids]
        cur_score = lp1[seq[0]] + lp2[seq[0], seq[1]]
        for k in range(n_len - 2):
            cur_score += lp3[seq[k], seq[k+1], seq[k+2]]
        
        best_r_score = cur_score
        best_r_perm = perm[:]
        
        t0, t1 = 4.5, 0.08
        for step in range(steps):
            i = rng.randrange(n_uniq)
            j = rng.randrange(26)
            if i == j:
                continue
            perm[i], perm[j] = perm[j], perm[i]
            
            seq = [perm[c] for c in c_ids]
            new_score = lp1[seq[0]] + lp2[seq[0], seq[1]]
            for k in range(n_len - 2):
                new_score += lp3[seq[k], seq[k+1], seq[k+2]]
            
            diff = new_score - cur_score
            if diff >= 0:
                cur_score = new_score
                if cur_score > best_r_score:
                    best_r_score = cur_score
                    best_r_perm = perm[:]
            else:
                temp = t0 * ((t1 / t0) ** (step / steps))
                if math.exp(diff / temp) > rng.random():
                    cur_score = new_score
                else:
                    perm[i], perm[j] = perm[j], perm[i]
        
        # Polish with exhaustive 2-opt hill climbing
        perm = best_r_perm[:]
        cur_score = best_r_score
        improved = True
        while improved:
            improved = False
            for i in range(n_uniq):
                for j in range(i + 1, 26):
                    perm[i], perm[j] = perm[j], perm[i]
                    seq = [perm[c] for c in c_ids]
                    ns = lp1[seq[0]] + lp2[seq[0], seq[1]]
                    for k in range(n_len - 2):
                        ns += lp3[seq[k], seq[k+1], seq[k+2]]
                    if ns > cur_score + 1e-9:
                        cur_score = ns
                        improved = True
                    else:
                        perm[i], perm[j] = perm[j], perm[i]
        
        if cur_score > best_overall_score:
            best_overall_score = cur_score
            best_overall_map = perm[:]
            
    plaintext = "".join(chr(65 + best_overall_map[c]) for c in c_ids)
    return best_overall_score, plaintext

# ==============================================================================
# 4. DORABELLA CIPHER & TRANSPOSITIONS
# ==============================================================================
LINES = [
    "A2 E3 B2 A3 A1 C2 G1 A3 D1 H2 B3 F2 F1 B1 F2 C3 F2 F2 C2 E3 E3 F2 B1 H1 H2 H1 C1 B3 F3".split(),
    "G1 F2 G1 C2 H1 A3 D1 D2 A3 B2 F2 F2 B2 C2 C1 F1 G1 F2 B3 F2 C2 G2 F3 F1 B1 H1 D1 D1 H1 B3 F3".split(),
    "B2 F3 C2 G2 F3 B2 B1 G2 G3 C1 F3 B2 F2 C2 G2 F1 F3 C1 A3 E3 C1 F3 C2 A3 B1 H1 A3".split()
]

def get_transpositions():
    l0, l1, l2 = LINES
    T = {}
    T["T0_Normal (L1->L3, G->D)"] = l0 + l1 + l2
    T["T1_Inverse_180 (L3->L1, D->G)"] = list(reversed(l0 + l1 + l2))
    T["T2_Lignes_inversees (L1->L3, D->G)"] = list(reversed(l0)) + list(reversed(l1)) + list(reversed(l2))
    T["T3_Boustrophedon_A (->, <-, ->)"] = l0 + list(reversed(l1)) + l2
    T["T4_Boustrophedon_B (<-, ->, <-)"] = list(reversed(l0)) + l1 + list(reversed(l2))
    
    cols_left = []
    for c in range(max(len(l0), len(l1), len(l2))):
        for l in (l0, l1, l2):
            if c < len(l):
                cols_left.append(l[c])
    T["T5_Colonnes_gauche (haut->bas)"] = cols_left
    
    cols_right = []
    max_len = 31
    padded = [([None]*(max_len - len(l)) + l) for l in (l0, l1, l2)]
    for c in range(max_len):
        for p in padded:
            if p[c] is not None:
                cols_right.append(p[c])
    T["T6_Colonnes_droite (haut->bas)"] = cols_right
    
    cols_zig = []
    for c in range(max(len(l0), len(l1), len(l2))):
        order = (l0, l1, l2) if c % 2 == 0 else (l2, l1, l0)
        for l in order:
            if c < len(l):
                cols_zig.append(l[c])
    T["T7_Colonnes_zigzag (vert.)"] = cols_zig

    T["T8_Lignes_bas_haut (L3,L2,L1 G->D)"] = l2 + l1 + l0
    return T

CONTROLS = {
    "EN": clean_text("Dear Miss Penny I cannot thank you enough for the pleasant days we spent at your rectory in July and I hope we meet again soon"),
    "IT": clean_text("Cara signorina Dora vi ringrazio di cuore per i giorni felici trascorsi insieme nella vostra bella casa e spero di rivedervi presto"),
    "DE": clean_text("Liebe Dora ich danke Ihnen sehr fuer die schoenen Tage in Ihrem Hause und hoffe dass wir bald wieder zusammen musizieren koennen")
}

def main():
    lms = {
        "EN": LangModel("English", CORPUS_EN),
        "IT": LangModel("Italiano", CORPUS_IT),
        "DE": LangModel("Deutsch", CORPUS_DE),
    }
    print("=== CORPORA ===")
    for k, lm in lms.items():
        print(f"  {k}: {lm.corpus_len} letters")

    rng = random.Random(1234)
    ctrl_scores = {}
    rand_scores = {}
    base_dora = LINES[0] + LINES[1] + LINES[2]

    print("\n=== CALIBRATION (CONTROLE POSITIF 87 CAR. vs MELANGE ALEATOIRE DE DORABELLA) ===")
    for lang, lm in lms.items():
        pt = CONTROLS[lang][:87]
        true_score = lm.score_text(pt)
        alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        sh = alpha[:]
        rng.shuffle(sh)
        enc_map = dict(zip(alpha, sh))
        ct = [enc_map[c] for c in pt]
        
        sc_pos, dec_pos = solve_masc(ct, lm, restarts=12, steps=6000, seed=42)
        acc = sum(1 for a, b in zip(pt, dec_pos) if a == b) / 87.0
        
        neg_sc = []
        for s_i in range(3):
            sh_d = base_dora[:]
            rng.shuffle(sh_d)
            sc_n, _ = solve_masc(sh_d, lm, restarts=8, steps=5000, seed=100+s_i)
            neg_sc.append(sc_n)
        
        ctrl_scores[lang] = (true_score, sc_pos, acc, dec_pos)
        rand_scores[lang] = (min(neg_sc), sum(neg_sc)/len(neg_sc), max(neg_sc))
        print(f"\n[{lang}] Vrai texte 87 car. (score reel = {true_score:.1f}):")
        print(f"  Original : {pt}")
        print(f"  Solveur  : {dec_pos}  (score = {sc_pos:.1f}, exactitude = {100*acc:.1f}%)")
        print(f"  Temoins aleatoires Dorabella (3 shuffles) : moy = {rand_scores[lang][1]:.1f} (max = {rand_scores[lang][2]:.1f})")

    transps = get_transpositions()
    print("\n=== ATTAQUE DES 9 TRANSPOSITIONS x 3 LANGUES ===")
    results = []
    for lang, lm in lms.items():
        print(f"\n--- LANGUE : {lang} (Temoin positif={ctrl_scores[lang][1]:.1f} | Hasard moy={rand_scores[lang][1]:.1f}, max={rand_scores[lang][2]:.1f}) ---")
        for tname, seq in transps.items():
            sc, dec = solve_masc(seq, lm, restarts=12, steps=6000, seed=777)
            z_gap = sc - rand_scores[lang][1]
            results.append((lang, tname, sc, z_gap, dec))
            print(f"  {tname:35s} | score={sc:6.1f} (vs hasard {z_gap:+5.1f}) | {dec[:60]}...")

    print("\n=== TOP 6 MEILLEURS CANDIDATS (ECART VS HASARD DE SA LANGUE) ===")
    results.sort(key=lambda x: -x[3])
    for lang, tname, sc, gap, dec in results[:6]:
        print(f"  [{lang}] {tname:35s} | score={sc:.1f} (ecart hasard={gap:+.1f} | pos={ctrl_scores[lang][1]:.1f})")
        print(f"       -> {dec}")

if __name__ == "__main__":
    main()
