"""Template-based generator that expands the training set (incl. negation: 'saru nathi' = negative)."""
import random
R = random.Random(7)
TAIL = ["", "", " yaar", " bhai", " 😊", " !!", " ho", " really"]
SUBJ = ["aa movie", "aa food", "aa phone", "tamari service", "aa song", "aa app", "aa hotel", "delivery", "aa match", "aa laptop", "aaje no din", "this camera"]
POS = ["bahu saras che", "ekdum mast che", "khub saru che", "superb che", "love it", "bov maja aavi", "best che", "jabardast che", "bahu gamyu", "awesome hatu", "kharab nathi", "not bad at all", "paisa vasool che", "very good", "bahut accha hai"]
NEG = ["bahu kharab che", "bakvas che", "worst che", "bov bekar che", "disappointed chu", "waste of money", "bilkul gamyu nahi", "saru nathi", "good nathi", "not good", "bahut bekar hai", "never again", "boring hatu", "bahu irritating che", "bakwas hai"]
NEU = ["kal khule che", "5 vagye start thase", "ahmedabad ma che", "ketlu price che", "monday thi chalu thase", "share karo", "number 4521 che", "kya malse", "vishe info joie che", "ke nahi khabar nathi"]
N = ["order", "phone", "food", "delivery", "ticket", "booking", "service", "app", "bill", "parcel"]
INT = {
 "greeting": ["kem cho", "hello", "hi bhai", "good morning", "namaste", "su chale che", "kem chho mitro", "hey kem cho", "jai shree krishna", "kaise ho"],
 "question": ["{n} kyare che", "{n} ketlu che", "{n} kya malse", "shu {n} available che", "how much is {n}", "{n} kem kaam kare che", "{n} kab aayega", "what about {n}"],
 "request": ["please {n} moklo", "{n} share karo", "plz {n} ni detail aapo", "mane {n} joie che", "can you send {n}", "{n} mate help karo", "{n} bhej do please"],
 "complaint": ["{n} bahu kharab che", "{n} kaam nathi karyu", "{n} late thayu", "{n} ma problem che", "worst {n} experience", "{n} nu refund nathi malyu", "{n} bilkul bekar hai"],
 "appreciation": ["{n} bahu saru che", "thank you {n} mast hatu", "{n} jabardast che", "love this {n}", "{n} ekdum best che", "tamari {n} ni maja aavi", "{n} bahut accha hai"]}
TOP = {
 "sports": ["cricket", "match", "india team", "ipl", "football", "kohli", "score", "wicket", "stadium", "world cup"],
 "entertainment": ["movie", "film", "song", "garba", "web series", "trailer", "actor", "bollywood", "gujarati natak", "dayro"],
 "food": ["dhokla", "fafda", "thali", "pizza", "chai", "restaurant", "khaman", "biryani", "ice cream", "undhiyu"],
 "technology": ["phone", "laptop", "wifi", "app", "software", "update", "internet", "charger", "ai chatbot", "android"],
 "politics": ["election", "sarkar", "minister", "vote", "budget", "party", "chunav", "government", "mla", "neta"]}
TT = ["{x} aaje bahu saras hatu", "{x} kyare che", "{x} vishe vaat karo", "mane {x} bahu gamyu", "{x} ma problem che", "{x} ni news joyi?", "aaje {x} ma shu thayu", "{x} nu update aavyu che", "best {x} kyu che"]

def gen():
    D = {"sentiment": {}, "intent": {}, "topic": {}}
    for lab, pool in (("positive", POS), ("negative", NEG), ("neutral", NEU)):
        D["sentiment"][lab] = list({f"{R.choice(SUBJ)} {R.choice(pool)}{R.choice(TAIL)}" for _ in range(400)})
    for lab, ts in INT.items():
        D["intent"][lab] = list({R.choice(ts).format(n=R.choice(N)) + R.choice(TAIL) for _ in range(250)})
    for lab, ws in TOP.items():
        D["topic"][lab] = list({R.choice(TT).format(x=R.choice(ws)) + R.choice(TAIL) for _ in range(250)})
    return D
