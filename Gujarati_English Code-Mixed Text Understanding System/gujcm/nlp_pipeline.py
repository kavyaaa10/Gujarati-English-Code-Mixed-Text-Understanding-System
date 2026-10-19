"""Gujarati-English(-Hindi) code-mixed text pipeline:
word-level language ID -> normalization -> transliteration."""
import re

GU_RANGE, HI_RANGE = (0x0A80, 0x0AFF), (0x0900, 0x097F)

# ---------- small lexicons (extend these for better accuracy) ----------
GU = set("""kem cho chho che chhe chhu chhi chhiye shu su sho maja majama saru sara saras bahu bov khub khoob
nathi tame tamne tamari tamaru hu mane mara mari maru amne amara apne aapne aavjo aavo jao jaao ghare ghar
mitra mitro dost bhai ben kaam aaje aaj pachi hamna have jovu joie joiye game gamyu gami thayu thai thay
karyu karo kari karvu kya kyare kevu kevi ketlu ane ke pan to j ma thi ni no nu na ne mate sathe vagye
dur najik malyu aavtu aave jaun jaish gayo gayi gayu hatu hati hato rahu raho rehshe moklo moklu khule
bakvas kharab sundar ekdum mast jabardast""".split())
HI = set("""hai hain kya nahi nahin bahut accha acha yaar mat kaise kyun kyu mera meri tum tumhara aap hum kar
raha rahi abhi bohot sab bilkul mujhe toh lekin magar zindagi pyaar bekar badhiya kaam nhi haan""".split())
EN = set("""the a an is are was were be been am i you he she it we they this that these those my your our their
and or but not no yes ok okay so very too really just only also with without for from to in on at of by
movie film song phone camera food restaurant service delivery order refund network traffic weather
office meeting exam bank station train shop time table address number price cost best worst good bad
great awesome nice super superb worst love loved hate like enjoy happy sad angry boring annoying
disappointed experience please thanks thank sorry hello hi hey congratulations again never always
late early waste money cheating full free share send call message match cricket game team win won lost
player score goal laptop app software internet wifi online update hang hangs login bug server
pizza burger dinner lunch breakfast tea coffee taste tasty spicy cold hot irritating enjoy""".split())

GU |= set("""kai koi badha badhu thodu vadhare ochhu saras sundar gamtu nahi aavyo aavyu gayu jovo joyu joyo jamvanu nasto
chai dudh pani rasto gaadi gadi shaher gaam sarkar chutni maate sathe vachche andar bahar upar niche aagal pachhal
tamaro tamaru tamara mari tari taru tari amaro khushi dukh gusso hasvu radvu vaat vato sambhal jo juo vaachvu lakhvu
bolvu bolo bol sambhalo ubhu betho besi besvu aavshe jashe karshe thashe hashe chale chalo chalo""".split())
HI |= set("""kuch koi sabhi bahot thoda zyada jyada kam mast zabardast sahi galat theek thik dost bhai ghar kaisa kaisi
kaise kab kahan kaun kitna kitne wala wali ke ki ka se me mein par aur ya bhi hi na mat ho hua hui gaya gayi karna
dekho dekh suno bolo chalo aao jao acchi accha achha pasand bakwas""".split())
EN |= set("""what when where who why how which can could would should will shall do does did have has had get got make made
take took give gave see saw go went come came know think want need help work works working try tried use used
news update country sports player india world cricket football election vote budget minister government
chatbot charger stadium trailer actor review rating price available buy sell cheap costly expensive fast slow
today tomorrow yesterday morning evening night day week month year""".split())

SLANG = {"thx": "thanks", "pls": "please", "plz": "please", "bt": "but", "gr8": "great", "u": "you",
         "r": "are", "ur": "your", "msg": "message", "nc": "nice", "gud": "good", "gd": "good",
         "bhut": "bahut", "bhot": "bahut", "bau": "bahu", "bov": "bahu", "bohot": "bahut",
         "che": "chhe", "se": "chhe", "chhey": "chhe", "chee": "chhe", "cho": "chho",
         "su": "shu", "sho": "shu", "kemm": "kem", "sru": "saru", "sara": "saru", "saaru": "saru",
         "maza": "maja", "mazza": "maja", "mja": "maja", "nthi": "nathi", "nai": "nathi",
         "khoob": "khub", "kyu": "kyun", "nhi": "nahi", "nahin": "nahi", "acha": "accha", "aaj": "aaje"}

def _script(tok):
    c = ord(tok[0])
    if GU_RANGE[0] <= c <= GU_RANGE[1]: return "GU_SCRIPT"
    if HI_RANGE[0] <= c <= HI_RANGE[1]: return "HI_SCRIPT"
    return None

def tokenize(text):
    return re.findall(r"[\u0A80-\u0AFF]+|[\u0900-\u097F]+|@\w+|#\w+|[A-Za-z0-9']+|[^\sA-Za-z0-9]", text)

def normalize_word(w):
    w = w.lower()
    cands = [w, re.sub(r"(.)\1{2,}", r"\1\1", w), re.sub(r"(.)\1+", r"\1", w)]  # soooo -> soo -> so
    for c in cands:
        if c in SLANG: return SLANG[c]
        if c in GU or c in HI or c in EN: return c
    return cands[1]

# ---------- word-level language identification ----------
GU_SUFFIX = ("no", "ni", "nu", "ne", "ma", "thi", "chhe", "vu", "yu", "shu", "ish", "aav")
def identify_language(tok, norm):
    if not re.search(r"\w", tok) or tok[0] in "@#": return "OTHER"
    if tok.isdigit(): return "OTHER"
    s = _script(tok)
    if s == "GU_SCRIPT": return "GU"
    if s == "HI_SCRIPT": return "HI"
    if norm in GU or norm == "chhe": return "GU"
    if norm in EN: return "EN"
    if norm in HI: return "HI"
    if re.search(r"chh|aa|bh|dh|kh|gh", norm) or (norm.endswith(GU_SUFFIX) and len(norm) > 3): return "GU"
    return "EN"

# ---------- rule-based Roman -> Gujarati transliteration ----------
CONS = {"chh": "છ", "kh": "ખ", "gh": "ઘ", "ch": "ચ", "jh": "ઝ", "th": "થ", "dh": "ધ", "ph": "ફ", "bh": "ભ",
        "sh": "શ", "k": "ક", "g": "ગ", "j": "જ", "t": "ત", "d": "દ", "n": "ન", "p": "પ", "b": "બ", "m": "મ",
        "y": "ય", "r": "ર", "l": "લ", "v": "વ", "w": "વ", "s": "સ", "h": "હ", "f": "ફ", "z": "ઝ", "c": "ક", "q": "ક"}
VOW = {"aa": ("આ", "ા"), "ai": ("ઐ", "ૈ"), "au": ("ઔ", "ૌ"), "ee": ("ઈ", "ી"), "ii": ("ઈ", "ી"), "oo": ("ઊ", "ૂ"),
       "a": ("અ", ""), "i": ("ઇ", "િ"), "u": ("ઉ", "ુ"), "e": ("એ", "ે"), "o": ("ઓ", "ો")}
DICT = {"chhe": "છે", "chho": "છો", "chhu": "છું", "shu": "શું", "kem": "કેમ", "maja": "મજા", "saru": "સારું",
        "bahu": "બહુ", "nathi": "નથી", "hu": "હું", "tame": "તમે", "mane": "મને", "ane": "અને", "ke": "કે", "pan": "પણ"}

def transliterate(word):
    w = word.lower()
    if w in DICT: return DICT[w]
    out, i = [], 0
    while i < len(w):
        v = next((w[i:i+n] for n in (2, 1) if w[i:i+n] in VOW), None)
        if v:                                    # standalone vowel at word start
            out.append(VOW[v][0]); i += len(v); continue
        c = next((w[i:i+n] for n in (3, 2, 1) if w[i:i+n] in CONS), None)
        if not c: out.append(w[i]); i += 1; continue
        out.append(CONS[c]); i += len(c)
        v = next((w[i:i+n] for n in (2, 1) if w[i:i+n] in VOW), None)
        if v: out.append(VOW[v][1]); i += len(v)
        elif i < len(w): out.append("્")         # halant for consonant clusters
    return "".join(out)

def analyze(text):
    toks = []
    for t in tokenize(text):
        isw = bool(re.match(r"[A-Za-z']", t))
        n = normalize_word(t) if isw else t
        lang = identify_language(t, n)
        known = (not isw) or n in GU or n in HI or n in EN or n in DICT
        tr = transliterate(n) if lang in ("GU", "HI") and re.match(r"[a-z]", n) else None
        toks.append({"token": t, "normalized": n, "lang": lang, "gujarati": tr,
                     "confidence": 1.0 if known else 0.6})   # 0.6 = heuristic guess (not in lexicon)
    real = [x["lang"] for x in toks if x["lang"] != "OTHER"]
    cmi = 0.0
    if real:
        cmi = round(100 * (1 - max(real.count(l) for l in set(real)) / len(real)), 1)
    switches = sum(1 for a, b in zip(real, real[1:]) if a != b)
    dist = {l: real.count(l) for l in ("GU", "HI", "EN")}
    return {"tokens": toks, "cmi": cmi, "switches": switches, "distribution": dist,
            "normalized_text": " ".join(x["normalized"] for x in toks)}
